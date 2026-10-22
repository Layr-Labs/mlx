// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/metal_kernel.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/fenced_input.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

#include <fmt/format.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlx::core::fast {

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // silence some warnings
  (void)is_precompiled_;
  (void)shared_memory_;

  const auto [tx, ty, tz] = threadgroup_;
  const auto [gx, gy, gz] = grid_;
  if (tx <= 0 || ty <= 0 || tz <= 0 || gx < 0 || gy < 0 || gz < 0) {
    throw std::invalid_argument("Invalid custom kernel launch dimensions.");
  }
  size_t tg_size = 1;
  for (auto dimension : {tx, ty, tz}) {
    if (static_cast<size_t>(dimension) >
        std::numeric_limits<std::uint32_t>::max() / tg_size) {
      throw std::invalid_argument(
          "Custom kernel threadgroup size exceeds 32 bits.");
    }
    tg_size *= static_cast<size_t>(dimension);
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  // The device limit is fixed: query it once rather than per launch.
  static const auto device_max = d.mtl_device()->maxThreadsPerThreadgroup();
  if (static_cast<size_t>(tx) > device_max.width ||
      static_cast<size_t>(ty) > device_max.height ||
      static_cast<size_t>(tz) > device_max.depth) {
    throw std::invalid_argument(
        "Custom kernel threadgroup exceeds device dimensions.");
  }

  for (auto i : mutable_inputs_) {
    if (i < 0 || i >= inputs.size()) {
      throw std::invalid_argument("Invalid mutable kernel input index.");
    }
    if (ensure_row_contiguous_ && !inputs[i].flags().row_contiguous) {
      throw std::invalid_argument(
          "Mutable kernel inputs cannot use an implicit contiguous copy.");
    }
  }

  // Waits on fenced inputs that Fence::wait_in_consumer gave to this kernel.
  // An input that is copied first is read before the kernel: wait for it now.
  std::vector<metal::FencedInput> fenced;
  if (!fenced_inputs_.empty()) {
    fenced = metal::take_fenced_inputs(this);
    for (auto it = fenced.begin(); it != fenced.end();) {
      auto& in = inputs[it->input];
      if (ensure_row_contiguous_ && !in.flags().row_contiguous) {
        metal::encode_fence_wait(s, in, *it);
        it = fenced.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::vector<array> copies;

  for (auto& out : outputs) {
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  // Bind the inputs themselves unless one needs a contiguous copy (the same
  // arrays either way, without a reference-counted copy of each).
  bool needs_copy = false;
  if (ensure_row_contiguous_) {
    for (const array& in : inputs) {
      needs_copy |= !in.flags().row_contiguous;
    }
  }
  std::vector<array> copied_inputs;
  if (needs_copy) {
    copied_inputs.reserve(inputs.size());
    for (const array& in : inputs) {
      copied_inputs.push_back(check_input(in));
    }
  }
  const std::vector<array>& checked_inputs =
      needs_copy ? copied_inputs : inputs;

  // Hashing the whole generated source (often tens of KB) on every dispatch
  // dominated the host cost of decode-sized custom kernels; the kernel
  // builder precomputes the identical name once per generated source.
  const std::string& source = source_->source;
  std::string derived_name;
  if (source_->library_name.empty()) {
    derived_name = fmt::format(
        "{}_{:x}_{}",
        name_,
        std::hash<std::string>{}(source),
        compile_options_);
  }
  const std::string& lib_name =
      source_->library_name.empty() ? derived_name : source_->library_name;
  // Launches that share a generated source skip the library and kernel
  // lookups (a name hash and a shared lock each) after the first one.
  std::call_once(source_->resolved, [&] {
    auto lib = d.get_library(lib_name, compile_options_, [&source] {
      return metal::utils() + source;
    });
    auto pipeline = d.get_kernel(name_, lib);
    source_->max_total_threads = pipeline->maxTotalThreadsPerThreadgroup();
    source_->pipeline = std::shared_ptr<void>(pipeline->retain(), [](void* p) {
      static_cast<MTL::ComputePipelineState*>(p)->release();
    });
  });
  auto kernel =
      static_cast<MTL::ComputePipelineState*>(source_->pipeline.get());
  auto max_tg_size = source_->max_total_threads;
  const bool retry_geometry = (tx == 512 || tx == 1024) && ty == 1 && tz == 1 &&
      gx >= tx && gx % tx == 0 && gy > 0 && gz > 0;
  if (tg_size > max_tg_size && retry_geometry) {
    const std::string declaration = "[[kernel]] void " + name_ + "(";
    const auto position = source.find(declaration);
    if (position != std::string::npos &&
        source.find(declaration, position + declaration.size()) ==
            std::string::npos) {
      // Keep the dispatch geometry and give the compiler its required size.
      auto constrained_source = source;
      constrained_source.insert(
          position,
          "[[max_total_threads_per_threadgroup(" + std::to_string(tg_size) +
              ")]] ");
      auto constrained_name =
          lib_name + "_required_tg_" + std::to_string(tg_size);
      auto constrained_lib = d.get_library(
          constrained_name, compile_options_, [&constrained_source] {
            return metal::utils() + constrained_source;
          });
      kernel = d.get_kernel(name_, constrained_lib);
      max_tg_size = kernel->maxTotalThreadsPerThreadgroup();
    }
  }
  if (tg_size > max_tg_size) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup (" << max_tg_size
        << "). Kernel: " << name_ << "; threadgroup=(" << tx << "," << ty << ","
        << tz << "); grid=(" << gx << "," << gy << "," << gz << ").";
    throw std::invalid_argument(msg.str());
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    if (std::find(mutable_inputs_.begin(), mutable_inputs_.end(), i) !=
        mutable_inputs_.end()) {
      compute_encoder.register_output_array(in);
    }
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (std::get<0>(shape_info)) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (std::get<1>(shape_info)) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (std::get<2>(shape_info)) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }
  for (int i : fenced_inputs_) {
    auto it = std::find_if(
        fenced.begin(), fenced.end(), [i](auto& f) { return f.input == i; });
    if (it != fenced.end()) {
      metal::bind_fenced_input(compute_encoder, *it, index);
    } else {
      // No CPU-stream producer in this eval: a count of 0 does not wait.
      compute_encoder.set_input_array(checked_inputs[i], index);
      compute_encoder.set_bytes(uint32_t(0), index + 1);
    }
    index += 2;
  }
  if (!fenced.empty()) {
    // As Fence::wait: start after every pass and kernel encoded before, so
    // the CPU work that the kernel waits for never waits for the kernel.
    compute_encoder.wait_for_all_pending();
    compute_encoder.barrier();
  }

  MTL::Size group_dims =
      MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  compute_encoder.add_temporaries(std::move(copies));
}

} // namespace mlx::core::fast
