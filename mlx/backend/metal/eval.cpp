// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_sets());
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_global_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_sets());
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(
        encoder.get_command_buffer(), arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  // Keep the inputs and siblings alive until the command buffer completes;
  // the output is skipped when an input donated its buffer to it. Retention
  // is batched per command buffer (`retain_until_completed`): a hash set plus
  // a Metal completion handler per primitive dominated host-side encode time
  // for decode-sized kernels. Retained AFTER eval_gpu, into the encoder's
  // current buffer: a primitive that synchronizes mid-eval (e.g. the
  // expert-tile route's descriptor retract check) commits and replaces the
  // buffer, and the current one holds the tail of this primitive's work,
  // which is exactly what the inputs must outlive. A duplicate input is
  // retained twice, which only costs one more reference.
  auto* out_data = arr.data_shared_ptr().get();
  for (auto& in : arr.inputs()) {
    if (in.data_shared_ptr().get() != out_data) {
      encoder.retain_until_completed(in.data_shared_ptr());
    }
  }
  for (auto& sib : arr.siblings()) {
    if (sib.data_shared_ptr().get() != out_data) {
      encoder.retain_until_completed(sib.data_shared_ptr());
    }
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s]() { scheduler::notify_task_completion(s); });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
  if (is_main_thread()) {
    metal::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
