// Copyright © 2024 Apple Inc.
#include "mlx/fence.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core {

struct FenceImpl {
  FenceImpl(Stream stream) {
    auto d = metal::device(stream.device).mtl_device();
    if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
      use_fast = false;
    } else if (__builtin_available(macOS 15, iOS 18, *)) {
      use_fast = env::metal_fast_synch();
    }

    if (!use_fast) {
      event = std::make_unique<Event>(stream);
    } else {
      // The timestamp, then the fence_update_coherent arrival count.
      auto buf = allocator::malloc(2 * sizeof(uint32_t)).ptr();
      fence = static_cast<void*>(buf);
      cpu_value()[0] = 0;
      cpu_value()[1] = 0;
    }
  }

  ~FenceImpl() {
    if (use_fast) {
      allocator::free(allocator::Buffer{static_cast<MTL::Buffer*>(fence)});
    }
  }
  bool use_fast{false};
  uint32_t count{0};
  // Threadgroups that fence_update_coherent counted in (mod 2^32).
  uint32_t arrivals{0};
  void* fence;
  std::unique_ptr<Event> event;

  std::atomic_uint* cpu_value() {
    return static_cast<std::atomic_uint*>(
        static_cast<MTL::Buffer*>(fence)->contents());
  }
};

Fence::Fence(Stream stream) {
  fence_ = std::make_shared<FenceImpl>(stream);
}

void Fence::wait(Stream stream, const array& x) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (!f.use_fast) {
    f.event->wait(stream);
    return;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, count = f.count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      while (f.cpu_value()[0] < count) {
      }
    });
    return;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  // Register outputs to ensure that no kernels which depends on the
  // output starts before this one is done
  compute_encoder.register_output_array(x);
  // The spin must not start ahead of GPU work encoded before it: without
  // hazard tracking Metal may run this pass before an earlier pass still in
  // flight, and when that pass is what the awaited CPU work depends on (a
  // GPU -> CPU -> GPU chain, e.g. a CPU-stream collective) the GPU spins
  // forever. Waiting on every pending pass costs nothing here: the CPU work
  // this spin waits for already follows them.
  compute_encoder.wait_for_all_pending();

  auto kernel = d.get_kernel("fence_wait");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(f.count, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  // The fence buffer must outlive the pass that spins on it.
  compute_encoder.retain_until_completed(fence_);
}

void Fence::update(Stream stream, const array& x, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;

  if (!f.use_fast) {
    f.event->set_value(f.count);
    f.event->signal(stream);
    return;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, count = f.count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      f.cpu_value()[0] = count;
    });
    return;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  if (cross_device && env::metal_fused_fence_update()) {
    uint32_t words = (x.data_size() * x.itemsize() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    constexpr uint32_t threads = 1024;
    constexpr uint32_t words_per_group = threads * 8;
    uint32_t groups =
        std::max<uint32_t>(1, (words + words_per_group - 1) / words_per_group);
    uint32_t target = 0;
    if (groups > 1) {
      f.arrivals += groups;
      target = f.arrivals;
    }
    auto kernel = d.get_kernel("fence_update_coherent");
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_bytes(words, 1);
    auto buf = static_cast<MTL::Buffer*>(f.fence);
    compute_encoder.set_buffer(buf, 2);
    compute_encoder.set_bytes(f.count, 3);
    compute_encoder.set_buffer(buf, 4, sizeof(uint32_t));
    compute_encoder.set_bytes(target, 5);
    // After every earlier kernel, as fence_update. This also keeps two
    // updates of this fence from overlapping, which `arrivals` requires.
    compute_encoder.barrier();
    compute_encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));
    compute_encoder.retain_until_completed(fence_);
    return;
  }

  // Launch input visibility kernels
  if (cross_device) {
    auto kernel = d.get_kernel("input_coherent");
    uint32_t nthreads = (x.data_size() * x.itemsize() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    MTL::Size group_dims = MTL::Size(1024, 1, 1);
    MTL::Size grid_dims = MTL::Size((nthreads + 1024 - 1) / 1024, 1, 1);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_bytes(nthreads, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Barrier on previous kernels
  compute_encoder.barrier();

  // Launch value update kernel
  auto kernel = d.get_kernel("fence_update");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(f.count, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  // The fence buffer must outlive the pass that writes it.
  compute_encoder.retain_until_completed(fence_);
}

} // namespace mlx::core
