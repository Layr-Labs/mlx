// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

// Inputs with fewer elements are passed in the constant address space
constexpr int max_constant_array_size = 8;

Stream resolve_metal_kernel_stream(StreamOrDevice s) {
  if (metal::is_available()) {
    // Default an unspecified stream to the GPU rather than the global default
    // device. Other inputs go through to_stream so explicit streams, devices
    // and ThreadLocalStream are handled as before.
    auto stream = std::holds_alternative<std::monostate>(s)
        ? default_stream(Device::gpu)
        : to_stream(s);
    if (stream.device != Device::gpu) {
      throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
    }
    return stream;
  }
  if (!detail::in_export_tracing()) {
    throw std::runtime_error("[metal_kernel] No Metal back-end.");
  }
  // Exporting without Metal: the kernel cannot run here but it can still be
  // recorded in the graph on a placeholder GPU stream. The importing process
  // remaps it to one of its own streams.
  auto* device = std::get_if<Device>(&s);
  auto* device_type = std::get_if<Device::DeviceType>(&s);
  auto* stream = std::get_if<Stream>(&s);
  auto* tl_stream = std::get_if<ThreadLocalStream>(&s);
  if ((device && *device != Device::gpu) ||
      (device_type && *device_type != Device::gpu) ||
      (stream && stream->device != Device::gpu) ||
      (tl_stream && tl_stream->device != Device::gpu)) {
    throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
  }
  return Stream(-1, Device::gpu);
}

// MLX_FENCED_INPUT_WAIT(name): thread 0 of each threadgroup spins until the
// CPU fence reaches the count, as fence_wait does. A count of 0 does not wait.
constexpr const char* fenced_input_wait = R"(
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
inline void mlx_fenced_input_wait(
    volatile coherent(system) device uint* fence,
    uint count,
    uint index) {
  constexpr metal::thread_scope scope =
      static_cast<metal::thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
  if (count == 0) {
    return;
  }
  if (index == 0) {
    while (true) {
      metal::atomic_thread_fence(
          metal::mem_flags::mem_device, metal::memory_order_seq_cst, scope);
      if (fence[0] >= count) {
        break;
      }
    }
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_device);
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device, metal::memory_order_seq_cst, scope);
}
#define MLX_FENCED_INPUT_WAIT(name) mlx_fenced_input_wait( \
    name##_fence, name##_fence_count, thread_index_in_threadgroup)
)";

constexpr const char* fenced_input_no_wait =
    "\n#define MLX_FENCED_INPUT_WAIT(name) ((void)0)\n";

bool fenced_inputs_enabled() {
#ifdef __APPLE__
  if (__builtin_available(macOS 15, iOS 18, *)) {
    return env::metal_fenced_inputs() && env::metal_fast_synch() &&
        !detail::in_export_tracing();
  }
#endif
  return false;
}

std::string write_signature(
    std::string func_name,
    const std::string& header,
    const std::string& source,
    const std::vector<std::string>& input_names,
    const std::vector<array>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<std::pair<std::string, TemplateArg>>& template_args,
    const std::vector<std::string>& attributes,
    const std::vector<std::tuple<bool, bool, bool>>& shape_infos,
    bool atomic_outputs,
    const std::vector<int>& mutable_inputs,
    const char* fence_preamble,
    const std::vector<int>& fenced_inputs) {
  std::string kernel_source;
  kernel_source.reserve(header.size() + source.size() + 16384);
  kernel_source += header;
  kernel_source += fence_preamble;
  // Auto-generate a function signature based on `template_args`
  // and the dtype/shape of the arrays passed as `inputs`.
  if (!template_args.empty()) {
    kernel_source += "template <";
    int i = 0;
    for (const auto& [name, arg] : template_args) {
      std::string param_type;
      if (std::holds_alternative<int>(arg)) {
        param_type = "int";
      } else if (std::holds_alternative<bool>(arg)) {
        param_type = "bool";
      } else if (std::holds_alternative<Dtype>(arg)) {
        param_type = "typename";
      }
      if (i > 0) {
        kernel_source += ", ";
      }
      kernel_source += param_type;
      kernel_source += " ";
      kernel_source += name;
      i++;
    }
    kernel_source += ">\n";
  }
  kernel_source += "[[kernel]] void ";
  kernel_source += func_name;
  kernel_source += "(\n";

  int index = 0;
  // Add inputs
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    bool is_mutable =
        std::find(mutable_inputs.begin(), mutable_inputs.end(), i) !=
        mutable_inputs.end();
    auto dtype = get_type_string(arr.dtype());
    std::string location = !is_mutable && arr.size() < max_constant_array_size
        ? "constant"
        : "device";
    std::string ref = !is_mutable && arr.ndim() == 0 ? "&" : "*";
    kernel_source += is_mutable ? "  " : "  const ";
    kernel_source += location;
    kernel_source += " ";
    kernel_source += dtype;
    kernel_source += ref;
    kernel_source += " ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]],\n";
    index++;
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (std::get<0>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int* " + name + "_shape [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<1>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int64_t* " + name + "_strides [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<2>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int& " + name + "_ndim [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
    }
  }
  // Add outputs
  for (int i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source += "  device ";
    auto type_string = get_type_string(dtype);
    if (atomic_outputs) {
      kernel_source += "atomic<";
    }
    kernel_source += type_string;
    if (atomic_outputs) {
      kernel_source += ">";
    }
    kernel_source += "* ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]]";
    if (index < inputs.size() + output_names.size() - 1 ||
        attributes.size() > 0) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }
  // Fenced inputs: the CPU fence and its count (attributes follow).
  for (int i : fenced_inputs) {
    const auto& name = input_names[i];
    kernel_source += "  volatile coherent(system) device uint* " + name +
        "_fence [[buffer(" + std::to_string(index++) + ")]],\n";
    kernel_source += "  constant uint& " + name + "_fence_count [[buffer(" +
        std::to_string(index++) + ")]],\n";
  }

  index = 0;
  for (const auto& attr : attributes) {
    kernel_source += attr;
    if (index < attributes.size() - 1) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }
  kernel_source += source;
  kernel_source += "\n}\n";
  return kernel_source;
}

std::string write_template(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  std::ostringstream template_def;
  template_def << "<";
  int i = 0;
  for (const auto& [name, arg] : template_args) {
    if (i > 0) {
      template_def << ", ";
    }
    if (std::holds_alternative<int>(arg)) {
      template_def << std::get<int>(arg);
    } else if (std::holds_alternative<bool>(arg)) {
      template_def << std::get<bool>(arg);
    } else if (std::holds_alternative<Dtype>(arg)) {
      template_def << get_type_string(std::get<Dtype>(arg));
    }
    i++;
  }
  template_def << ">";
  return template_def.str();
}

std::string make_template_hash(const std::string& template_def) {
  std::string template_hash;
  template_hash.reserve(template_def.size());
  for (size_t i = 0; i < template_def.size(); ++i) {
    auto c = template_def[i];
    if (c == '<' || c == '>') {
      template_hash += '_';
    } else if (
        c == ',' && i + 1 < template_def.size() && template_def[i + 1] == ' ') {
      template_hash += '_';
      ++i;
    } else {
      template_hash += c;
    }
  }
  template_hash.pop_back();
  return template_hash;
}

} // namespace

CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header /* = "" */,
    bool ensure_row_contiguous /* = true */,
    bool atomic_outputs /* = false */,
    const CompileOptions& compile_options /* = {} */) {
  return metal_kernel_with_mutable_inputs(
      name,
      input_names,
      output_names,
      source,
      {},
      header,
      ensure_row_contiguous,
      atomic_outputs,
      compile_options);
}

CustomKernelFunction metal_kernel_with_mutable_inputs(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::vector<std::string>& mutable_input_names,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs,
    const CompileOptions& compile_options) {
  return metal_kernel_with_fenced_inputs(
      name,
      input_names,
      output_names,
      source,
      mutable_input_names,
      {},
      header,
      ensure_row_contiguous,
      atomic_outputs,
      compile_options);
}

CustomKernelFunction metal_kernel_with_fenced_inputs(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::vector<std::string>& mutable_input_names,
    const std::vector<std::string>& fenced_input_names,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs,
    const CompileOptions& compile_options) {
  std::vector<int> mutable_inputs;
  for (const auto& mutable_name : mutable_input_names) {
    auto it = std::find(input_names.begin(), input_names.end(), mutable_name);
    if (it == input_names.end() ||
        std::count(input_names.begin(), input_names.end(), mutable_name) != 1) {
      throw std::invalid_argument("Mutable input must name one kernel input.");
    }
    int index = static_cast<int>(it - input_names.begin());
    if (std::find(mutable_inputs.begin(), mutable_inputs.end(), index) !=
        mutable_inputs.end()) {
      throw std::invalid_argument("Duplicate mutable kernel input.");
    }
    mutable_inputs.push_back(index);
  }
  std::sort(mutable_inputs.begin(), mutable_inputs.end());
  std::vector<int> fenced_inputs;
  for (const auto& fenced_name : fenced_input_names) {
    auto it = std::find(input_names.begin(), input_names.end(), fenced_name);
    if (it == input_names.end() ||
        std::count(input_names.begin(), input_names.end(), fenced_name) != 1) {
      throw std::invalid_argument("Fenced input must name one kernel input.");
    }
    int index = static_cast<int>(it - input_names.begin());
    if (std::find(fenced_inputs.begin(), fenced_inputs.end(), index) !=
            fenced_inputs.end() ||
        std::find(mutable_inputs.begin(), mutable_inputs.end(), index) !=
            mutable_inputs.end()) {
      throw std::invalid_argument(
          "A fenced kernel input must be unique and not mutable.");
    }
    for (const auto& suffix : {"_fence", "_fence_count"}) {
      auto param = fenced_name + suffix;
      if (std::count(input_names.begin(), input_names.end(), param) ||
          std::count(output_names.begin(), output_names.end(), param)) {
        throw std::invalid_argument(
            "Kernel argument name " + param + " is used by a fenced input.");
      }
    }
    if (source.find("MLX_FENCED_INPUT_WAIT(" + fenced_name + ")") ==
        std::string::npos) {
      throw std::invalid_argument(
          "The kernel source must call MLX_FENCED_INPUT_WAIT(" + fenced_name +
          ") before it reads the fenced input.");
    }
    fenced_inputs.push_back(index);
  }
  std::sort(fenced_inputs.begin(), fenced_inputs.end());
  if (output_names.empty()) {
    throw std::invalid_argument(
        "[metal_kernel] Must specify at least one output.");
  }
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  for (auto& n : input_names) {
    std::tuple<bool, bool, bool> shape_info;
    std::get<0>(shape_info) = source.find(n + "_shape") != std::string::npos;
    std::get<1>(shape_info) = source.find(n + "_strides") != std::string::npos;
    std::get<2>(shape_info) = source.find(n + "_ndim") != std::string::npos;
    shape_infos.push_back(shape_info);
  }
  const std::vector<std::pair<std::string, std::string>> metal_attributes = {
      {"dispatch_quadgroups_per_threadgroup", "uint"},
      {"dispatch_simdgroups_per_threadgroup", "uint"},
      {"dispatch_threads_per_threadgroup", "uint3"},
      {"grid_origin", "uint3"},
      {"grid_size", "uint3"},
      {"quadgroup_index_in_threadgroup", "uint"},
      {"quadgroups_per_threadgroup", "uint"},
      {"simdgroup_index_in_threadgroup", "uint"},
      {"simdgroups_per_threadgroup", "uint"},
      {"thread_execution_width", "uint"},
      {"thread_index_in_quadgroup", "uint"},
      {"thread_index_in_simdgroup", "uint"},
      {"thread_index_in_threadgroup", "uint"},
      {"thread_position_in_grid", "uint3"},
      {"thread_position_in_threadgroup", "uint3"},
      {"threadgroup_position_in_grid", "uint3"},
      {"threadgroups_per_grid", "uint3"},
      {"threads_per_grid", "uint3"},
      {"threads_per_simdgroup", "uint"},
      {"threads_per_threadgroup", "uint3"},
  };

  std::vector<std::string> attributes;
  for (const auto& [attr, dtype] : metal_attributes) {
    if (source.find(attr) != std::string::npos ||
        (!fenced_inputs.empty() && attr == "thread_index_in_threadgroup")) {
      attributes.push_back("  " + dtype + " " + attr + " [[" + attr + "]]");
    }
  }

  // The generated source is a pure function of this kernel object (header,
  // body, names, attributes, shape infos, atomic and mutable inputs: all
  // fixed here) and of the per-call key built below: the template
  // arguments and, per input, its dtype and how it is passed (mutable,
  // scalar, constant or device; a 0-d mutable input also drops its shape
  // parameters), the output dtypes and whether fenced waits are compiled in.
  // Cache the kernel name, source and Metal library name per key, so a
  // repeated call neither formats names nor regenerates the (often multi-KB)
  // source, and every launch of one entry shares it (and its pipeline, see
  // CustomKernel::eval_gpu). Shared by every copy of the returned function.
  struct Generated {
    std::string kernel_name;
    std::shared_ptr<const CustomKernelSource> source;
  };
  struct SourceCache {
    std::mutex mtx;
    std::unordered_map<std::string, std::shared_ptr<const Generated>> entries;
  };
  auto source_cache = std::make_shared<SourceCache>();

  return [=,
          shape_infos = std::move(shape_infos),
          attributes = std::move(attributes)](
             const std::vector<array>& inputs,
             const std::vector<Shape>& output_shapes,
             const std::vector<Dtype>& output_dtypes,
             std::tuple<int, int, int> grid,
             std::tuple<int, int, int> threadgroup,
             const std::vector<std::pair<std::string, TemplateArg>>&
                 template_args = {},
             std::optional<float> init_value = std::nullopt,
             bool verbose = false,
             StreamOrDevice s_ = {}) {
    auto check_size = [](size_t actual, size_t expected, const char* name) {
      if (actual == expected) {
        return;
      }
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `" << name << "` to have size "
          << expected << " but got size " << actual << "." << std::endl;
      throw std::invalid_argument(msg.str());
    };
    check_size(inputs.size(), input_names.size(), "inputs");
    check_size(output_shapes.size(), output_names.size(), "output_shapes");
    check_size(output_dtypes.size(), output_names.size(), "output_dtypes");

    auto s = resolve_metal_kernel_stream(s_);
    const bool fenced = !fenced_inputs.empty() && fenced_inputs_enabled();

    // The per-call key as raw bytes: each template argument (name, kind,
    // value), each input's dtype and passing, each output's dtype and the
    // fenced form. It determines the kernel name and the source below.
    thread_local std::string key;
    key.clear();
    for (const auto& [arg_name, arg] : template_args) {
      key += arg_name;
      key += '\0';
      key += static_cast<char>(arg.index());
      if (auto* v = std::get_if<int>(&arg)) {
        key.append(reinterpret_cast<const char*>(v), sizeof(int));
      } else if (auto* b = std::get_if<bool>(&arg)) {
        key += *b ? '1' : '0';
      } else {
        key += static_cast<char>(std::get<Dtype>(arg).val());
      }
    }
    key += '\0';
    for (int i = 0; i < inputs.size(); ++i) {
      const auto& arr = inputs[i];
      key += static_cast<char>(arr.dtype().val());
      if (std::find(mutable_inputs.begin(), mutable_inputs.end(), i) !=
          mutable_inputs.end()) {
        key += arr.ndim() == 0 ? 'z' : 'n';
      } else if (arr.ndim() == 0) {
        key += 's';
      } else if (arr.size() < max_constant_array_size) {
        key += 'c';
      } else {
        key += 'd';
      }
    }
    key += '\0';
    for (const auto& dtype : output_dtypes) {
      key += static_cast<char>(dtype.val());
    }
    key += fenced ? 'f' : 'u';

    std::shared_ptr<const Generated> generated;
    {
      std::lock_guard lock(source_cache->mtx);
      if (auto it = source_cache->entries.find(key);
          it != source_cache->entries.end()) {
        generated = it->second;
      }
    }
    if (!generated) {
      std::string kernel_name = "custom_kernel_" + name;
      std::string template_def = "";
      if (!template_args.empty()) {
        template_def = write_template(template_args);
        auto template_hash = make_template_hash(template_def);
        kernel_name += "_";
        kernel_name += template_hash;
      }

      // The generated source depends on the dtypes of the inputs and outputs
      // and on how each input is passed (see `write_signature`). Include them
      // in the kernel name so that a given name always maps to the same
      // source.
      for (int i = 0; i < inputs.size(); ++i) {
        const auto& arr = inputs[i];
        kernel_name += "_";
        kernel_name += get_type_string(arr.dtype());
        if (std::find(mutable_inputs.begin(), mutable_inputs.end(), i) !=
            mutable_inputs.end()) {
          kernel_name += "m";
        } else if (arr.ndim() == 0) {
          kernel_name += "s";
        } else if (arr.size() < max_constant_array_size) {
          kernel_name += "c";
        }
      }
      for (const auto& dtype : output_dtypes) {
        kernel_name += "_";
        kernel_name += get_type_string(dtype);
      }

      std::string kernel_source = write_signature(
          kernel_name,
          header,
          source,
          input_names,
          inputs,
          output_names,
          output_dtypes,
          template_args,
          attributes,
          shape_infos,
          atomic_outputs,
          mutable_inputs,
          fenced_inputs.empty() ? ""
              : fenced          ? fenced_input_wait
                                : fenced_input_no_wait,
          fenced ? fenced_inputs : std::vector<int>{});

      if (!template_args.empty()) {
        auto instantiation = kernel_name + template_def;
        kernel_source += "\ntemplate [[host_name(\"";
        kernel_source += kernel_name;
        kernel_source += "\")]] [[kernel]] decltype(";
        kernel_source += instantiation;
        kernel_source += ") ";
        kernel_source += instantiation;
        kernel_source += ";\n";
      }
      // Exactly the name CustomKernel::eval_gpu derived before.
      auto library_name = fmt::format(
          "{}_{:x}_{}",
          kernel_name,
          std::hash<std::string>{}(kernel_source),
          compile_options.serialize());
      auto entry = std::make_shared<Generated>();
      entry->kernel_name = std::move(kernel_name);
      entry->source = std::make_shared<const CustomKernelSource>(
          std::move(kernel_source), std::move(library_name));
      generated = std::move(entry);
      std::lock_guard lock(source_cache->mtx);
      source_cache->entries.emplace(key, generated);
    }

    if (verbose) {
      std::cout << "Generated source code for `" << name << "`:" << std::endl
                << "```" << std::endl
                << generated->source->source << std::endl
                << "```" << std::endl;
    }

    if (!mutable_inputs.empty()) {
      return array::make_arrays(
          output_shapes,
          output_dtypes,
          std::make_shared<MutableInputCustomKernel>(
              s,
              generated->kernel_name,
              generated->source,
              grid,
              threadgroup,
              shape_infos,
              ensure_row_contiguous,
              init_value,
              std::vector<ScalarArg>{},
              false,
              0,
              compile_options.serialize(),
              mutable_inputs,
              fenced ? fenced_inputs : std::vector<int>{}),
          inputs);
    }
    return array::make_arrays(
        output_shapes,
        output_dtypes,
        std::make_shared<CustomKernel>(
            s,
            generated->kernel_name,
            generated->source,
            grid,
            threadgroup,
            shape_infos,
            ensure_row_contiguous,
            init_value,
            std::vector<ScalarArg>{},
            false,
            0,
            compile_options.serialize(),
            std::vector<int>{},
            fenced ? fenced_inputs : std::vector<int>{}),
        inputs);
  };
}

} // namespace mlx::core::fast
