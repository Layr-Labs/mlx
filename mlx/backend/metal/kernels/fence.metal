// Copyright © 2024 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to update timestamp
[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// input_coherent and fence_update in one dispatch. The last threadgroup to
// finish its words writes the timestamp. `arrivals` only increases: this
// dispatch moves it from target - groups to target (mod 2^32), so an
// unfinished earlier dispatch can stop the timestamp but never release it
// early. One threadgroup does not use `arrivals`.
[[kernel]] void fence_update_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    volatile coherent(system) device uint* timestamp [[buffer(2)]],
    constant uint& value [[buffer(3)]],
    device metal::atomic_uint* arrivals [[buffer(4)]],
    constant uint& target [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  for (uint i = group * threads + index; i < size; i += groups * threads) {
    input[i] = input[i];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
  metal::threadgroup_barrier(metal::mem_flags::mem_device);
  if (index != 0) {
    return;
  }
  if (groups > 1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    uint previous = metal::atomic_fetch_add_explicit(
        arrivals, 1u, metal::memory_order_relaxed);
    if (previous + 1u != target) {
      return;
    }
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to spin wait for timestamp value
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    if (timestamp[0] >= value) {
      break;
    }
  }
}
