// Copyright © 2026 Eigen Labs.
#include "mlx/backend/common/process_allocation_cap.h"
#include <Block.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include "mlx/backend/common/allocation_footprint.h"

using namespace mlx::core::allocator;
using Cap = ProcessAllocationCap;
using Lease = ProcessBackingLease;

template <typename F>
void refuses(F f) {
  bool refused = false;
  try {
    f();
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused);
}
void invariant(const Cap& cap) {
  auto s = cap.snapshot();
  assert(
      s.charged_bytes ==
      s.heap_bytes + s.live_bytes + s.cached_bytes + s.pending_bytes +
          s.unreclaimed_bytes);
  assert(s.charged_bytes <= s.peak_bytes && s.peak_bytes <= s.limit_bytes);
}
std::atomic<int> releases{0};
bool unmap(void* p, size_t n) {
  ++releases;
  return munmap(p, n) == 0;
}

using Callback = void (^)();
Callback retainedCallback(std::shared_ptr<Lease> lease) {
  return Block_copy(^{
    lease->finish();
  });
}

int main() {
  const size_t page = static_cast<size_t>(getpagesize());
  {
    Cap c;
    refuses([&] { c.install(0, 0); });
    refuses([&] { c.install(3, 4); });
    assert(!c.snapshot().installed);
    c.install(16, 4);
    refuses([&] { c.install(32, 4); });
    c.reserve(12);
    refuses([&] { c.reserve(1); });
    c.failed_before_acquisition(12);
    assert(c.snapshot().charged_bytes == 4);
    invariant(c);
    Cap huge;
    huge.install(SIZE_MAX, SIZE_MAX - 1);
    huge.reserve(1);
    refuses([&] { huge.reserve(SIZE_MAX); });
    huge.failed_before_acquisition(1);
    invariant(huge);
    refuses([&] { round_allocation_size(SIZE_MAX, page); });
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(4 * page, page);
    // Real VM backing, real copied Blocks ABI; no Metal/device initialization.
    for (int i = 0; i < 1000; ++i) {
      auto lease = std::make_shared<Lease>(c, page);
      void* p = mmap(
          nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
      assert(p != MAP_FAILED);
      lease->attach(p, unmap);
      auto callback = retainedCallback(lease);
      lease->acquired();
      lease->cache();
      lease->reuse();
      lease->retire();
      lease.reset();
      assert(c->snapshot().unreclaimed_bytes == page);
      assert(c->snapshot().charged_bytes == 2 * page);
      callback();
      callback(); // defensive idempotence; unmap exactly once
      Block_release(callback);
      assert(c->snapshot().charged_bytes == page);
      invariant(*c);
    }
    assert(
        releases == 1000); // Churn does not accumulate a fake high-water cap.
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(page, 0);
    auto lease = std::make_shared<Lease>(c, page);
    auto callback = retainedCallback(lease);
    lease->acquired();
    lease->retire();
    lease.reset();
    refuses(
        [&] { Lease next(c, 1); }); // retained Metal equivalent blocks refund
    std::thread completion([&] { callback(); });
    completion.join();
    Block_release(callback);
    Lease next(c, page); // delayed completion makes capacity reusable
    invariant(*c);
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(page, 0);
    {
      Lease failed(c, page);
    } // no VM acquired
    assert(c->snapshot().charged_bytes == 0);
    auto failed = std::make_shared<Lease>(c, page);
    void* p = mmap(
        nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(p != MAP_FAILED);
    failed->attach(p, unmap);
    failed->finish(); // Metal constructor returned nil after VM acquisition
    failed.reset();
    assert(c->snapshot().charged_bytes == 0 && releases == 1001);
    Lease quarantine(c, page);
    quarantine.attach(nullptr, [](void*, size_t) { return false; });
    quarantine.acquired();
    quarantine.retire();
    quarantine.finish();
    assert(c->snapshot().unreclaimed_bytes == page);
    refuses([&] { Lease next(c, 1); });
    invariant(*c);
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(page, 0);
    // Foreign pages stay caller-owned even after callback and ledger refund.
    auto* p = static_cast<unsigned char*>(mmap(
        nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    assert(p != MAP_FAILED);
    p[0] = 42;
    {
      Lease wrapper(c, page);
      wrapper.acquired();
      wrapper.retire();
      wrapper.finish();
    }
    assert(p[0] == 42 && c->snapshot().charged_bytes == 0);
    assert(munmap(p, page) == 0);
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(1, 0);
    std::atomic<int> arrived{0}, successes{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
      threads.emplace_back([&] {
        std::shared_ptr<Lease> lease;
        try {
          lease = std::make_shared<Lease>(c, 1);
          ++successes;
        } catch (...) {
        }
        ++arrived;
        while (!release) {
          std::this_thread::yield();
        }
      });
    }
    while (arrived != 2) {
      std::this_thread::yield();
    }
    assert(successes == 1 && c->snapshot().pending_bytes == 1);
    release = true;
    for (auto& t : threads) {
      t.join();
    }
    assert(c->snapshot().charged_bytes == 0);
    invariant(*c);
  }
  {
    auto c = std::make_shared<Cap>();
    c->install(page, 0);
    std::weak_ptr<Cap> weak = c;
    auto lease = std::make_shared<Lease>(c, page);
    auto callback = retainedCallback(lease);
    lease->acquired();
    lease->retire();
    lease.reset();
    c.reset();
    assert(!weak.expired());
    callback();
    Block_release(callback);
    assert(weak.expired()); // callback owns ledger, not an allocator stack
                            // reference
  }
  std::cout
      << "PASS: install/overflow, 1000 VM churn cycles, delayed copied block, "
         "failure rollback/quarantine, foreign ownership, concurrent reserve, ledger lifetime\n";
}
