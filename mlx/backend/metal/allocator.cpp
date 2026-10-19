// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/memory.h"

#include <mach/vm_page_size.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace mlx::core {

constexpr size_t resource_options =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

namespace allocator {

Allocator& allocator() {
  return metal::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr_);
  assert(buf->storageMode() != MTL::StorageModePrivate);
  return buf->contents();
}

bool can_reuse_alien_buffer(void* ptr) {
  if (!ptr) {
    return true;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr);
  return buf->storageMode() != MTL::StorageModePrivate;
}

} // namespace allocator

namespace metal {

MetalAllocator::MetalAllocator(Device& d)
    : device_(d.mtl_device()),
      residency_sets_(d.residency_sets()),
      buffer_cache_(
          vm_page_size,
          [](MTL::Buffer* buf) { return buf->length(); },
          [this](MTL::Buffer* buf) {
            relinquish_capped(buf, true);
            if (!buf->heap()) {
              residency_sets_.erase(buf);
            }
            auto pool = metal::new_scoped_memory_pool();
            buf->release();
          }) {
  const auto& info = gpu::device_info(0);
  auto memsize = std::get<size_t>(info.at("memory_size"));
  auto max_rec_size =
      std::get<size_t>(info.at("max_recommended_working_set_size"));
  resource_limit_ = std::get<size_t>(info.at("resource_limit"));
  // Optional override (Darkbloom): MLX_RESOURCE_LIMIT lets an operator/test pin
  // the Metal resource-COUNT ceiling below the OS default (~499000). Used to
  // deterministically exercise the count-aware high-water trim, and as a safety
  // valve to force earlier cache reclamation on a box seeing the resource-limit
  // crash. The value may only LOWER the ceiling (it is clamped to the OS limit)
  // — raising it above what the hardware/OS reports would invite the very crash
  // this guards against. Strictly validated: a plain unsigned decimal that
  // consumes the whole string, is non-zero, and does not overflow; anything
  // else (empty, sign, junk, range error) is ignored and the OS limit stands.
  if (const char* rl = std::getenv("MLX_RESOURCE_LIMIT")) {
    while (*rl == ' ' || *rl == '\t') {
      ++rl;
    }
    if (*rl >= '0' && *rl <= '9') { // unsigned decimal only (reject sign/junk)
      errno = 0;
      char* end = nullptr;
      unsigned long long v = std::strtoull(rl, &end, 10);
      bool consumed_all = end != rl && *end == '\0';
      if (consumed_all && errno != ERANGE && v > 0 &&
          v <= std::numeric_limits<size_t>::max()) {
        resource_limit_ = std::min(static_cast<size_t>(v), resource_limit_);
      }
    }
  }
  block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);
  gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);
  max_pool_size_ = block_limit_;
  bool is_vm = std::get<std::string>(info.at("device_name")) ==
      "Apple Paravirtual device";
  if (is_vm) {
    return;
  }
  auto pool = metal::new_scoped_memory_pool();
  auto heap_desc = MTL::HeapDescriptor::alloc()->init()->autorelease();
  heap_desc->setResourceOptions(resource_options);
  heap_desc->setSize(heap_size_);
  heap_ = NS::TransferPtr(device_->newHeap(heap_desc));
  residency_sets_.insert(heap_.get());
}

MetalAllocator::~MetalAllocator() = default;

size_t MetalAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
};

size_t MetalAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, block_limit_);
  gc_limit_ = std::min(
      block_limit_,
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()));
  return limit;
};

MemorySnapshot MetalAllocator::get_memory_snapshot() {
  std::lock_guard lock(mutex_);
  return {active_memory_, buffer_cache_.cache_size(), peak_memory_};
}

size_t MetalAllocator::get_memory_limit() {
  return block_limit_;
}

size_t MetalAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  residency_sets_.resize(wired_limit_);
  return limit;
};

void MetalAllocator::install_process_allocation_cap(size_t bytes) {
  std::unique_lock lock(mutex_);
  if (buffer_allocation_attempted_) {
    throw std::invalid_argument(
        "allocation cap requires a pristine buffer allocator");
  }
  cap_->install(bytes, heap_ ? heap_->size() : 0);
  cap_installed_ = true;
}

ProcessAllocationCapSnapshot
MetalAllocator::get_process_allocation_cap_snapshot() {
  return cap_->snapshot();
}

void MetalAllocator::relinquish_capped(MTL::Buffer* buffer, bool cached) {
  if (!cap_installed_) {
    return;
  }
  auto entry = capped_buffers_.find(buffer);
  if (entry == capped_buffers_.end()) {
    throw std::logic_error("untracked capped buffer");
  }
  if (entry->second) {
    entry->second->retire();
  }
  capped_buffers_.erase(entry);
}

Buffer MetalAllocator::malloc_capped(
    size_t size,
    std::unique_lock<std::mutex>& lock) {
  if (auto* cached = buffer_cache_.reuse_from_cache(size)) {
    if (auto lease = capped_buffers_.at(cached)) {
      lease->reuse();
    }
    active_memory_ += cached->length();
    peak_memory_ = std::max(peak_memory_, active_memory_);
    return Buffer{cached};
  }
  // Trim idle backing before refusing a new shape. Actual callbacks, not cache
  // eviction, return credit; externally retained buffers remain charged.
  auto snapshot = cap_->snapshot();
  const auto charge = allocator::round_allocation_size(size, vm_page_size);
  if (charge > device_->maxBufferLength()) {
    throw std::invalid_argument(
        "page-aligned backing exceeds Metal buffer limit");
  }
  if (charge > snapshot.limit_bytes - snapshot.charged_bytes ||
      num_resources_ >= resource_limit_ ||
      buffer_cache_.cache_size() > max_pool_size_) {
    num_resources_ -= buffer_cache_.clear();
  }
  if (num_resources_ >= resource_limit_) {
    throw std::runtime_error("Metal resource limit exceeded");
  }
  MTL::Buffer* buffer = nullptr;
  std::shared_ptr<allocator::ProcessBackingLease> lease;
  if (size < small_size_ && heap_) {
    buffer = heap_->newBuffer(size, resource_options);
  }
  if (!buffer) {
    // Reserve before releasing allocator lock, including in-flight allocations.
    lease = std::make_shared<allocator::ProcessBackingLease>(cap_, charge);
    ++num_resources_;
    lock.unlock();
    try {
      void* pages = mmap(
          nullptr,
          charge,
          PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANON,
          -1,
          0);
      if (pages == MAP_FAILED) {
        throw std::bad_alloc();
      }
      lease->attach(pages, [](void* p, size_t n) { return munmap(p, n) == 0; });
      // Fault every page from the CPU before wrapping it. Pages a no-copy
      // buffer first receives through the GPU are not charged to this
      // process's physical footprint, which hides the model from the OS
      // memory-pressure accounting and from Activity Monitor.
      for (size_t offset = 0; offset < charge; offset += vm_page_size) {
        static_cast<volatile char*>(pages)[offset] = 0;
      }
      buffer = device_->newBuffer(
          pages, charge, resource_options, ^(void*, NS::UInteger) {
            lease->finish();
          });
      if (!buffer) {
        throw std::runtime_error("Metal no-copy allocation failed");
      }
      lease->acquired();
    } catch (...) {
      if (buffer) {
        buffer->release();
      } else {
        lease->finish();
      }
      lock.lock();
      --num_resources_;
      throw;
    }
    lock.lock();
  } else {
    ++num_resources_;
  }
  try {
    capped_buffers_.emplace(buffer, lease);
    if (!buffer->heap()) {
      residency_sets_.insert(buffer);
    }
  } catch (...) {
    capped_buffers_.erase(buffer);
    if (lease) {
      lease->retire();
    }
    --num_resources_;
    buffer->release();
    throw;
  }
  active_memory_ += buffer->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);
  return Buffer{buffer};
}

Buffer MetalAllocator::malloc(size_t size) {
  // Metal doesn't like empty buffers
  if (size == 0) {
    return Buffer{nullptr};
  }

  // More helpful message if maximum buffer length is exceeded
  if (size > device_->maxBufferLength()) {
    std::ostringstream msg;
    msg << "[metal::malloc] Attempting to allocate " << size
        << " bytes which is greater than"
        << " the maximum allowed buffer size of " << device_->maxBufferLength()
        << " bytes.";
    throw std::runtime_error(msg.str());
  }

  // Align up memory
  if (size > vm_page_size) {
    size = allocator::round_allocation_size(size, vm_page_size);
  }

  // Try the cache
  std::unique_lock lk(mutex_);
  buffer_allocation_attempted_ = true;
  if (cap_installed_) {
    return malloc_capped(size, lk);
  }
  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    size_t mem_required = get_active_memory() + get_cache_memory() + size;

    // If we have a lot of memory pressure try to reclaim memory from the cache.
    // NOTE: release_cached_buffers takes a BYTES-to-free target; when the
    // buffers are tiny this frees only a few entries even though the COUNT is
    // the binding constraint, so the byte path alone cannot bound
    // num_resources_ (see the count-aware reclaim below).
    if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }

    // Count-aware reclaim (Darkbloom): the Metal resource COUNT limit
    // (resource_limit_, ~iogpu.rsrc_limit/499000) is independent of byte usage.
    // Under churn with many distinct buffer shapes (varied prompt lengths,
    // growing KV caches, multiple co-resident models) freed buffers are
    // recycled into the size-keyed cache and never reused at that exact size,
    // so the cache ENTRY COUNT creeps toward the limit while byte usage stays
    // modest — the byte-driven trim above never fires (its threshold is
    // ~physical RAM). Once the count crosses a high-water mark, proactively
    // clear the cache (pure reuse pool — clearing only costs re-allocation,
    // never correctness) so the count drops back to the live working set. This
    // makes the count limit unreachable by any request mix / batching method,
    // while the existing byte limits keep total memory below physical RAM.
    if (resource_limit_ > 0 &&
        num_resources_ >= (resource_limit_ * resource_high_water_num_) /
                resource_high_water_den_) {
      num_resources_ -= buffer_cache_.clear();
    }

    // Allocate new buffer if needed
    if (num_resources_ >= resource_limit_) {
      std::ostringstream msg;
      msg << "[metal::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
    lk.unlock();
    if (size < small_size_ && heap_) {
      buf = heap_->newBuffer(size, resource_options);
    }
    if (!buf) {
      buf = device_->newBuffer(size, resource_options);
    }
    if (!buf) {
      std::ostringstream msg;
      msg << "[malloc] Unable to allocate " << size << " bytes.";
      throw std::runtime_error(msg.str());
    }
    lk.lock();
    try {
      if (!buf->heap()) {
        residency_sets_.insert(buf);
      }
    } catch (...) {
      lk.unlock();
      buf->release();
      throw;
    }
    num_resources_++;
  }

  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);

  // Maintain the cache below the requested limit
  if (get_cache_memory() > max_pool_size_) {
    num_resources_ -= buffer_cache_.release_cached_buffers(
        get_cache_memory() - max_pool_size_);
  }

  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
    if (cap_installed_) {
      if (auto lease = capped_buffers_.at(buf)) {
        lease->cache();
      }
    }
  } else {
    relinquish_capped(buf, false);
    num_resources_--;
    if (!buf->heap()) {
      residency_sets_.erase(buf);
    }
    lk.unlock();
    auto pool = metal::new_scoped_memory_pool();
    buf->release();
  }
}

size_t MetalAllocator::size(Buffer buffer) const {
  return static_cast<MTL::Buffer*>(buffer.ptr())->length();
}

Buffer MetalAllocator::make_buffer(void* ptr, size_t size) {
  std::unique_lock lock(mutex_);
  buffer_allocation_attempted_ = true;
  if (!cap_installed_) {
    lock.unlock();
    auto buf = device_->newBuffer(ptr, size, resource_options, nullptr);
    if (!buf) {
      return Buffer{nullptr};
    }
    lock.lock();
    try {
      residency_sets_.insert(buf);
    } catch (...) {
      lock.unlock();
      buf->release();
      throw;
    }
    active_memory_ += buf->length();
    peak_memory_ = std::max(peak_memory_, active_memory_);
    num_resources_++;
    return Buffer{static_cast<void*>(buf)};
  }
  std::shared_ptr<allocator::ProcessBackingLease> lease;
  if (cap_installed_) {
    if (!size || size % vm_page_size ||
        reinterpret_cast<uintptr_t>(ptr) % vm_page_size || !ptr ||
        size > device_->maxBufferLength()) {
      throw std::invalid_argument(
          "capped foreign backing must be page aligned");
    }
    if (num_resources_ >= resource_limit_) {
      throw std::runtime_error("Metal resource limit exceeded");
    }
    lease = std::make_shared<allocator::ProcessBackingLease>(cap_, size);
  }
  ++num_resources_;
  lock.unlock();
  MTL::Buffer* buf = nullptr;
  try {
    if (lease) {
      // Charge wrapper lifetime; never free or assume ownership of caller
      // pages.
      buf = device_->newBuffer(
          ptr, size, resource_options, ^(void*, NS::UInteger) {
            lease->finish();
          });
    } else {
      buf = device_->newBuffer(ptr, size, resource_options, nullptr);
    }
    if (buf && lease) {
      lease->acquired();
    }
  } catch (...) {
    if (buf) {
      buf->release();
    } else if (lease) {
      lease->finish();
    }
    lock.lock();
    --num_resources_;
    throw;
  }
  lock.lock();
  if (!buf) {
    if (lease) {
      lease->finish();
    }
    --num_resources_;
    throw std::runtime_error("capped Metal no-copy wrapper allocation failed");
  }
  try {
    if (lease) {
      capped_buffers_.emplace(buf, lease);
    }
    residency_sets_.insert(buf);
  } catch (...) {
    if (lease) {
      capped_buffers_.erase(buf);
      lease->retire();
    }
    --num_resources_;
    buf->release();
    throw;
  }
  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);
  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::release(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  relinquish_capped(buf, false);
  num_resources_--;
  residency_sets_.erase(buf);
  lk.unlock();
  auto pool = metal::new_scoped_memory_pool();
  buf->release();
}

MetalAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of MetalAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static MetalAllocator* allocator_ =
      new MetalAllocator(device(mlx::core::Device::gpu));
  return *allocator_;
}

} // namespace metal

AllocationFootprintPolicy get_allocation_footprint_policy() noexcept {
  // Capped fallback buffers use whole VM pages, including subpage requests.
  return {vm_page_size, vm_page_size, vm_page_size, 0, vm_page_size};
}

size_t get_allocation_size_upper_bound(size_t size) {
  size_t result;
  if (!get_allocation_footprint_policy().upper_bound(size, result)) {
    throw std::overflow_error("allocation footprint overflow");
  }
  return result;
}

void install_process_allocation_cap(size_t bytes) {
  metal::allocator().install_process_allocation_cap(bytes);
}
ProcessAllocationCapSnapshot get_process_allocation_cap_snapshot() {
  return metal::allocator().get_process_allocation_cap_snapshot();
}

size_t set_cache_limit(size_t limit) {
  return metal::allocator().set_cache_limit(limit);
}
size_t set_memory_limit(size_t limit) {
  return metal::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return metal::allocator().get_memory_limit();
}
size_t set_wired_limit(size_t limit) {
  if (limit > std::get<size_t>(
                  gpu::device_info(0).at("max_recommended_working_set_size"))) {
    throw std::invalid_argument(
        "[metal::set_wired_limit] Setting a wired limit larger than "
        "the maximum working set size is not allowed.");
  }
  return metal::allocator().set_wired_limit(limit);
}
MemorySnapshot get_memory_snapshot() {
  return metal::allocator().get_memory_snapshot();
}
size_t get_active_memory() {
  return metal::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return metal::allocator().get_peak_memory();
}
void reset_peak_memory() {
  metal::allocator().reset_peak_memory();
}
size_t get_cache_memory() {
  return metal::allocator().get_cache_memory();
}
size_t get_num_resources() {
  return metal::allocator().get_num_resources();
}
size_t get_resource_limit() {
  return metal::allocator().get_resource_limit();
}
void clear_cache() {
  return metal::allocator().clear_cache();
}

} // namespace mlx::core
