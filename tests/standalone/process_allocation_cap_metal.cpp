// Copyright © 2026 Eigen Labs.
#include <mach/mach.h>
#include <objc/runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <string>
#include <thread>
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

using namespace mlx::core;
namespace a = mlx::core::allocator;
void require(bool value, const char* what) {
  if (!value) {
    throw std::runtime_error(what);
  }
}
template <typename F>
void refuses(F f) {
  bool refused = false;
  try {
    f();
  } catch (const std::exception&) {
    refused = true;
  }
  require(refused, "required refusal did not occur");
}
auto snapshot() {
  return get_process_allocation_cap_snapshot();
}
// The allocator makes no fixed heap on a virtual GPU, such as the one of a
// GitHub-hosted macOS runner (metal/allocator.cpp).
bool paravirtual() {
  return std::get<std::string>(gpu::device_info(0).at("device_name")) ==
      "Apple Paravirtual device";
}
void settled(size_t expected) {
  for (int i = 0; i < 200 && snapshot().charged_bytes != expected; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  require(
      snapshot().charged_bytes == expected, "callback credit did not settle");
}

// Process-local Metal API interception; no production failure switches.
IMP original_no_copy = nullptr;
IMP original_allocate = nullptr;
IMP original_residency = nullptr;
MTL::Buffer* retained_failure_buffer = nullptr;
bool reject_no_copy = false;
MTL::Buffer* intercept_no_copy(
    id self,
    SEL selector,
    void* p,
    NS::UInteger n,
    MTL::ResourceOptions opts,
    void (^cleanup)(void*, NS::UInteger)) {
  if (reject_no_copy) {
    reject_no_copy = false;
    return nullptr;
  }
  auto fn = reinterpret_cast<MTL::Buffer* (*)(id,
                                              SEL,
                                              void*,
                                              NS::UInteger,
                                              MTL::ResourceOptions,
                                              void (^)(void*, NS::UInteger))>(
      original_no_copy);
  auto* b = fn(self, selector, p, n, opts, cleanup);
  if (b) {
    b->retain();
    retained_failure_buffer = b;
  }
  return b;
}
MTL::Buffer* intercept_allocate(
    id self,
    SEL selector,
    NS::UInteger n,
    MTL::ResourceOptions opts) {
  auto fn = reinterpret_cast<
      MTL::Buffer* (*)(id, SEL, NS::UInteger, MTL::ResourceOptions)>(
      original_allocate);
  auto* b = fn(self, selector, n, opts);
  if (b) {
    b->retain();
    retained_failure_buffer = b;
  }
  return b;
}
MTL::ResidencySet*
fail_residency(id, SEL, MTL::ResidencySetDescriptor*, NS::Error**) {
  throw std::bad_alloc(); // insert has already added its map entry
}
struct Intercept {
  Method method;
  IMP previous;
  Intercept(MTL::Device* device, const char* selector, IMP replacement)
      : method(class_getInstanceMethod(
            object_getClass(reinterpret_cast<id>(device)),
            sel_registerName(selector))),
        previous(nullptr) {
    require(method != nullptr, "Metal interception selector missing");
    previous = method_setImplementation(method, replacement);
  }
  ~Intercept() {
    method_setImplementation(method, previous);
  }
};

struct PendingMetalConstruction {
  std::mutex mutex;
  std::condition_variable changed;
  size_t entered{0};
  bool complete{false};
  std::array<MTL::Buffer*, 2> retained{nullptr, nullptr};
  IMP original{nullptr};
};
PendingMetalConstruction* pending_construction = nullptr;
MTL::Buffer* intercept_pending(
    id self,
    SEL selector,
    void* p,
    NS::UInteger n,
    MTL::ResourceOptions opts,
    void (^cleanup)(void*, NS::UInteger)) {
  auto& state = *pending_construction;
  auto fn = reinterpret_cast<MTL::Buffer* (*)(id,
                                              SEL,
                                              void*,
                                              NS::UInteger,
                                              MTL::ResourceOptions,
                                              void (^)(void*, NS::UInteger))>(
      state.original);
  // Construct real Metal buffers before holding their return to the allocator.
  auto* buffer = fn(self, selector, p, n, opts, cleanup);
  std::unique_lock lock(state.mutex);
  const auto index = state.entered++;
  if (index < state.retained.size() && buffer) {
    buffer->retain();
    state.retained[index] = buffer;
  }
  state.changed.notify_all();
  if (index < state.retained.size()) {
    state.changed.wait(lock, [&] { return state.complete; });
  }
  return buffer;
}
void concurrent_case(size_t page) {
  auto& device = metal::device(Device::gpu);
  const size_t cap = 8 << 20;
  install_process_allocation_cap(cap);
  set_cache_limit(0);
  const auto heap = snapshot().heap_bytes;
  const auto chunk = (cap - heap) / 2;
  require(
      chunk % page == 0 && chunk * 2 + heap == cap,
      "nonintegral test envelope");
  PendingMetalConstruction state;
  pending_construction = &state;
  std::array<a::Buffer, 2> buffers{a::Buffer(nullptr), a::Buffer(nullptr)};
  std::array<std::exception_ptr, 2> errors;
  bool both_pending = false, contender_refused = false;
  ProcessAllocationCapSnapshot pending_snapshot;
  size_t actual_bytes = 0;
  a::Buffer contender(nullptr);
  {
    Intercept hook(
        device.mtl_device(),
        "newBufferWithBytesNoCopy:length:options:deallocator:",
        reinterpret_cast<IMP>(intercept_pending));
    state.original = hook.previous;
    std::array<std::thread, 2> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
      workers[i] = std::thread([&, i] {
        try {
          buffers[i] = a::allocator().malloc(chunk);
        } catch (...) {
          errors[i] = std::current_exception();
        }
      });
    }
    {
      std::unique_lock lock(state.mutex);
      both_pending = state.changed.wait_for(
          lock, std::chrono::seconds(5), [&] { return state.entered == 2; });
      if (both_pending && state.retained[0] && state.retained[1]) {
        actual_bytes =
            state.retained[0]->length() + state.retained[1]->length();
      }
    }
    pending_snapshot = snapshot();
    if (both_pending) {
      try {
        contender = a::allocator().malloc(page);
      } catch (const std::exception&) {
        contender_refused = true;
      }
    }
    {
      std::lock_guard lock(state.mutex);
      state.complete = true;
    }
    state.changed.notify_all();
    for (auto& worker : workers) {
      worker.join();
    }
  }
  pending_construction = nullptr;
  require(
      both_pending,
      "allocator did not allow two real in-flight Metal allocations");
  require(!errors[0] && !errors[1], "real Metal construction failed");
  require(actual_bytes + heap == cap, "unexpected actual Metal backing total");
  require(
      pending_snapshot.pending_bytes == 2 * chunk &&
          pending_snapshot.charged_bytes == cap &&
          pending_snapshot.live_bytes == 0,
      "in-flight Metal backing was not reserved before completion");
  require(
      contender_refused && !contender.ptr() && state.entered == 2,
      "contender reached Metal despite exhausted pending budget");
  require(
      snapshot().peak_bytes == cap && snapshot().pending_bytes == 0 &&
          snapshot().live_bytes == 2 * chunk,
      "pending-to-live transfer changed cap charge");
  for (auto buffer : buffers) {
    a::allocator().free(buffer);
  }
  clear_cache();
  require(
      snapshot().unreclaimed_bytes == 2 * chunk &&
          snapshot().charged_bytes == cap,
      "allocator release refunded externally retained Metal backing");
  refuses([&] { a::allocator().malloc(page); });
  state.retained[0]->release();
  settled(heap + chunk);
  auto recovered = a::allocator().malloc(chunk);
  require(
      snapshot().charged_bytes == cap && snapshot().peak_bytes == cap,
      "callback recovery failed exact-cap replacement");
  a::allocator().free(recovered);
  clear_cache();
  settled(heap + chunk);
  state.retained[1]->release();
  settled(heap);
  require(
      snapshot().pending_bytes == 0 && snapshot().unreclaimed_bytes == 0,
      "concurrent allocation left stranded charges");
}
void failure_case(const std::string& mode, size_t page) {
  const bool capped = mode.find("ordinary") == std::string::npos;
  const bool foreign = mode.find("foreign") != std::string::npos;
  auto& device = metal::device(Device::gpu);
  auto& sets = device.residency_sets();
  (void)snapshot(); // initialize allocator and its fixed heap
  if (capped) {
    install_process_allocation_cap(8 << 20);
  }
  const auto baseline = capped ? snapshot().charged_bytes : 0;
  set_cache_limit(0);
  if (mode == "nil-foreign") {
    void* p = mmap(
        nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require(p != MAP_FAILED, "foreign map failed");
    int deleter_calls = 0;
    bool threw = false;
    {
      Intercept hook(
          device.mtl_device(),
          "newBufferWithBytesNoCopy:length:options:deallocator:",
          reinterpret_cast<IMP>(intercept_no_copy));
      original_no_copy = hook.previous;
      reject_no_copy = true;
      try {
        array foreign_array(
            p, {static_cast<int>(page / sizeof(float))}, float32, [&](void*) {
              ++deleter_calls;
            });
      } catch (const std::exception&) {
        threw = true;
      }
    }
    require(munmap(p, page) == 0, "caller unmap failed");
    require(threw, "nil wrapper silently copied instead of throwing");
    require(deleter_calls == 0, "failed wrapper invoked caller deleter");
    settled(baseline);
    return;
  }
  set_wired_limit(8 << 20);
  sets.set_max_bytes_per_set(page);
  require(sets.enabled(), "residency sets are disabled on this device");
  // The failure is injected where a new residency set is made. Fill one set
  // with a page first, so that the next page needs a new set. On a device with
  // the fixed heap, set 0 holds the heap and the filler gets a new set; on a
  // device without it, the filler goes into the empty set 0.
  auto filler = a::allocator().malloc(page);
  const auto filled = capped ? snapshot().charged_bytes : 0;
  const auto wired = sets.wired_size();
  const auto resources = get_num_resources();
  const auto active = get_active_memory();
  void* foreign_pages = nullptr;
  if (foreign) {
    foreign_pages = mmap(
        nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require(foreign_pages != MAP_FAILED, "foreign map failed");
  }
  {
    Intercept no_copy(
        device.mtl_device(),
        "newBufferWithBytesNoCopy:length:options:deallocator:",
        reinterpret_cast<IMP>(intercept_no_copy));
    original_no_copy = no_copy.previous;
    Intercept allocated(
        device.mtl_device(),
        "newBufferWithLength:options:",
        reinterpret_cast<IMP>(intercept_allocate));
    original_allocate = allocated.previous;
    Intercept registration(
        device.mtl_device(),
        "newResidencySetWithDescriptor:error:",
        reinterpret_cast<IMP>(fail_residency));
    original_residency = registration.previous;
    refuses([&] {
      if (foreign) {
        (void)a::make_buffer(foreign_pages, page);
      } else {
        (void)a::allocator().malloc(page);
      }
    });
  }
  require(
      retained_failure_buffer != nullptr,
      "failure did not reach allocated buffer");
  // Keep the object alive to diagnose stale registration without dereferencing
  // freed memory. On the broken path resize wires this failed allocation.
  set_wired_limit((8 << 20) + page);
  require(
      sets.wired_size() == wired,
      "failed insert left a tracked Metal allocation");
  require(
      get_num_resources() == resources && get_active_memory() == active,
      "failed registration changed allocator counters");
  if (capped) {
    require(
        snapshot().charged_bytes == filled + page,
        "retained failure refunded early");
  }
  require(
      retained_failure_buffer->retainCount() == 1,
      "allocator leaked failed buffer reference");
  retained_failure_buffer->release();
  retained_failure_buffer = nullptr;
  if (capped) {
    settled(filled);
  }
  if (foreign) {
    require(munmap(foreign_pages, page) == 0, "foreign pages lost ownership");
  }
  a::allocator().free(filler);
  clear_cache();
  if (capped) {
    settled(baseline);
  }
}
// Capped backing is owned no-copy VM. Pages first written by the GPU must
// still be charged to this process, or the OS cannot see the model's memory.
size_t physical_footprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  require(
      task_info(
          mach_task_self(),
          TASK_VM_INFO,
          reinterpret_cast<task_info_t>(&info),
          &count) == KERN_SUCCESS,
      "task footprint unavailable");
  return static_cast<size_t>(info.phys_footprint);
}

bool footprint_growth_at_least(size_t before, size_t after, size_t expected) {
  return after >= before && after - before >= expected;
}

void footprint_delta_case() {
  require(
      !footprint_growth_at_least(1024, 1023, 512),
      "decrease wrapped into growth");
  require(
      !footprint_growth_at_least(1024, 1024, 1), "unchanged footprint grew");
  require(!footprint_growth_at_least(1024, 1535, 512), "short growth passed");
  require(footprint_growth_at_least(1024, 1536, 512), "exact growth failed");
  require(
      !footprint_growth_at_least(SIZE_MAX, 0, 1), "maximum baseline wrapped");
}

void gpu_footprint_case() {
  const size_t bytes = size_t(512) << 20;
  install_process_allocation_cap(size_t(2) << 30);
  const auto before = physical_footprint();
  {
    auto filled =
        full({int(bytes / sizeof(float))}, 1.0f, float32, Device::gpu);
    eval(filled);
    synchronize(default_stream(Device::gpu));
    const auto after = physical_footprint();
    require(
        footprint_growth_at_least(before, after, bytes - (bytes >> 4)),
        "GPU-written capped backing missing from process footprint");
  }
  clear_cache();
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "one fresh-process case required");
    const std::string mode = argv[1];
    const size_t page = static_cast<size_t>(getpagesize());
    // Facts of the device that decide which paths the cases take.
    std::cout << "device: "
              << std::get<std::string>(gpu::device_info(0).at("device_name"))
              << ", residency sets: "
              << (metal::device(Device::gpu).residency_sets().enabled()
                      ? "enabled"
                      : "disabled")
              << '\n';
    if (mode == "footprint-delta") {
      footprint_delta_case();
    } else if (mode == "gpu-footprint") {
      gpu_footprint_case();
    } else if (mode == "concurrent") {
      concurrent_case(page);
    } else if (mode.find("registration-") == 0 || mode == "nil-foreign") {
      failure_case(mode, page);
    } else if (mode == "late") {
      auto b = a::allocator().malloc(page);
      a::allocator().free(b);
      clear_cache();
      refuses([] { install_process_allocation_cap(8 << 20); });
      require(!snapshot().installed, "late install mutated state");
    } else {
      // The allocator's fixed heap is created when queried, before
      // installation.
      require(!snapshot().installed, "process already capped");
      const size_t cap = 8 << 20;
      install_process_allocation_cap(cap);
      const auto baseline = snapshot();
      const auto heap = baseline.heap_bytes;
      require((heap > 0) == !paravirtual(), "fixed heap does not match device");
      require(baseline.charged_bytes == heap, "existing heap not charged");
      require(baseline.limit_bytes == cap, "wrong cap");
      if (mode == "install") {
        refuses([&] { install_process_allocation_cap(cap); });
        refuses([&] { install_process_allocation_cap(cap + page); });
        require(snapshot().limit_bytes == cap, "limit changed");
        refuses([] { a::allocator().malloc(SIZE_MAX); });
        require(
            snapshot().charged_bytes == heap, "overflow leaked reservation");
        auto exact = a::allocator().malloc(cap - heap);
        require(snapshot().charged_bytes == cap, "exact fill mismatch");
        refuses([&] { a::allocator().malloc(page); });
        a::allocator().free(exact);
        clear_cache();
        settled(heap);
      } else if (mode == "retained") {
        auto b = a::allocator().malloc(cap - heap);
        auto* metal = static_cast<MTL::Buffer*>(b.ptr());
        metal->retain();
        a::allocator().free(b);
        clear_cache();
        require(
            snapshot().unreclaimed_bytes == cap - heap,
            "release refunded early");
        require(snapshot().charged_bytes == cap, "extra reference lost charge");
        refuses([&] { a::allocator().malloc(page); });
        metal->release();
        settled(heap);
        auto replacement = a::allocator().malloc(cap - heap);
        a::allocator().free(replacement);
        clear_cache();
        settled(heap);
      } else if (mode == "churn") {
        auto tiny = a::allocator().malloc(8);
        require(
            (static_cast<MTL::Buffer*>(tiny.ptr())->heap() != nullptr) ==
                (heap > 0),
            "fixed heap use does not match device");
        require(snapshot().charged_bytes == heap, "heap double charged");
        a::allocator().free(tiny);
        clear_cache();
        for (size_t i = 0; i < 1000; ++i) {
          const auto size = page * (1 + i % 31);
          auto b = a::allocator().malloc(size);
          const auto bytes = snapshot().charged_bytes;
          auto pointer = b.ptr();
          a::allocator().free(b);
          require(
              snapshot().cached_bytes > 0 && snapshot().charged_bytes == bytes,
              "cache transfer refunded credit");
          auto reused = a::allocator().malloc(size);
          require(
              reused.ptr() == pointer && snapshot().charged_bytes == bytes,
              "cache reuse allocated new backing");
          a::allocator().free(reused);
          clear_cache();
          settled(heap);
        }
        require(snapshot().peak_bytes < cap, "churn accumulated backing");
      } else if (mode == "foreign") {
        void* p = mmap(
            nullptr,
            page,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0);
        require(p != MAP_FAILED, "foreign map failed");
        static_cast<unsigned char*>(p)[0] = 42;
        auto b = a::make_buffer(p, page);
        require(b.ptr() != nullptr, "foreign wrapper failed");
        auto* metal = static_cast<MTL::Buffer*>(b.ptr());
        metal->retain();
        a::release(b);
        require(
            snapshot().charged_bytes == heap + page,
            "foreign wrapper refunded early");
        metal->release();
        settled(heap);
        require(
            static_cast<unsigned char*>(p)[0] == 42,
            "caller pages invalidated");
        refuses([&] { a::make_buffer(static_cast<char*>(p) + 1, page); });
        require(munmap(p, page) == 0, "caller unmap failed");
      } else if (mode == "gpu") {
        {
          // Real native graph roots stay owned through GPU evaluation and
          // readback.
          auto input = arange(1024, float32, Device::gpu);
          auto output =
              add(multiply(input, array(2.0f), Device::gpu),
                  array(3.0f),
                  Device::gpu);
          async_eval(output);
          synchronize(default_stream(Device::gpu));
          for (int i = 0; i < 1024; ++i) {
            require(
                output.data<float>()[i] == i * 2.0f + 3.0f,
                "GPU value mismatch");
          }
          require(snapshot().live_bytes > 0, "evaluated roots uncharged");
        }
        synchronize(default_stream(Device::gpu));
        clear_cache();
        settled(heap);
      } else {
        throw std::runtime_error("unknown case");
      }
      require(snapshot().pending_bytes == 0, "pending reservation leaked");
      require(snapshot().charged_bytes <= cap, "cap exceeded");
    }
    std::cout << "PASS " << mode << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
