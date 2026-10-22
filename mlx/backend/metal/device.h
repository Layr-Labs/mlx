// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/common/gemma4_expert_qmm.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/device.h"

namespace mlx::core::metal {

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

class Device;

// A set of buffer pointers for the encoder's hazard tracking. A small set is
// a vector with a linear search; a larger set also gets an open addressing
// index. clear() keeps the memory, so a warm encoder does not allocate.
class BufferSet {
 public:
  BufferSet() = default;
  BufferSet(const BufferSet&) = delete;
  BufferSet& operator=(const BufferSet&) = delete;

  // Returns true if `p` was not in the set.
  bool insert(const void* p) {
    if (!indexed_) {
      if (std::find(items_.begin(), items_.end(), p) != items_.end()) {
        return false;
      }
      items_.push_back(p);
      if (items_.size() > kLinearMax) {
        build_index();
      }
      return true;
    }
    auto i = find_slot(key(p));
    if (slots_[i] != kEmpty) {
      return false;
    }
    slots_[i] = key(p);
    items_.push_back(p);
    if (2 * items_.size() > slots_.size()) {
      build_index();
    }
    return true;
  }

  template <typename It>
  void insert(It first, It last) {
    for (; first != last; ++first) {
      insert(*first);
    }
  }

  bool contains(const void* p) const {
    if (!indexed_) {
      return std::find(items_.begin(), items_.end(), p) != items_.end();
    }
    return slots_[find_slot(key(p))] != kEmpty;
  }

  void erase(const void* p) {
    auto it = std::find(items_.begin(), items_.end(), p);
    if (it == items_.end()) {
      return;
    }
    *it = items_.back();
    items_.pop_back();
    if (indexed_) {
      erase_slot(find_slot(key(p)));
    }
  }

  // build_index() resets every slot, so the index needs no clearing here.
  void clear() {
    items_.clear();
    indexed_ = false;
  }

  void swap(BufferSet& other) noexcept {
    items_.swap(other.items_);
    slots_.swap(other.slots_);
    std::swap(indexed_, other.indexed_);
  }

  size_t size() const {
    return items_.size();
  }
  std::vector<const void*>::const_iterator begin() const {
    return items_.begin();
  }
  std::vector<const void*>::const_iterator end() const {
    return items_.end();
  }

 private:
  static constexpr size_t kLinearMax = 16;
  static constexpr uintptr_t kEmpty = ~uintptr_t(0);

  static uintptr_t key(const void* p) {
    return reinterpret_cast<uintptr_t>(p);
  }
  static size_t hash(uintptr_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return static_cast<size_t>(k);
  }
  // The slot that holds `k`, or the empty slot where it would go.
  size_t find_slot(uintptr_t k) const {
    size_t mask = slots_.size() - 1;
    size_t i = hash(k) & mask;
    while (slots_[i] != kEmpty && slots_[i] != k) {
      i = (i + 1) & mask;
    }
    return i;
  }
  // Remove slot i and move back later entries of its probe run.
  void erase_slot(size_t i) {
    size_t mask = slots_.size() - 1;
    for (size_t j = (i + 1) & mask; slots_[j] != kEmpty; j = (j + 1) & mask) {
      size_t home = hash(slots_[j]) & mask;
      bool stays = (i <= j) ? (i < home && home <= j) : (i < home || home <= j);
      if (!stays) {
        slots_[i] = slots_[j];
        i = j;
      }
    }
    slots_[i] = kEmpty;
  }
  void build_index() {
    size_t capacity = 64;
    while (capacity < 4 * items_.size()) {
      capacity *= 2;
    }
    slots_.assign(capacity, kEmpty);
    for (auto p : items_) {
      slots_[find_slot(key(p))] = key(p);
    }
    indexed_ = true;
  }

  std::vector<const void*> items_;
  std::vector<uintptr_t> slots_;
  bool indexed_{false};
};

class MLX_API CommandEncoder {
 public:
  CommandEncoder(Device& d, int index, ResidencySets& residency_sets);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent_ = true;
    }
    ~ConcurrentContext() {
      enc.concurrent_ = false;
      enc.prev_outputs_.insert(
          enc.concurrent_outputs_.begin(), enc.concurrent_outputs_.end());
      enc.concurrent_outputs_.clear();
    }

   private:
    CommandEncoder& enc;
  };

  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);

  void add_temporary(array arr);
  void add_temporaries(std::vector<array> arrays);

  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();

  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel) {
    get_command_encoder()->setComputePipelineState(kernel);
  }

  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, size_t nelems, int idx) {
    get_command_encoder()->setBytes(
        vec.data(), nelems * sizeof(typename Vec::value_type), idx);
  }
  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    return get_command_encoder()->setBytes(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    return get_command_encoder()->setBytes(&v, sizeof(T), idx);
  }

  void set_threadgroup_memory_length(size_t length, int idx) {
    get_command_encoder()->setThreadgroupMemoryLength(length, idx);
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  void barrier();

  // The current pass waits for every pass still in flight on this queue, not
  // only those that wrote its inputs (see Fence::wait).
  void wait_for_all_pending() {
    wait_all_pending_ = true;
  }

  void end_encoding();
  void wait_event(Event event, uint64_t value);
  void signal_event(Event event, uint64_t value);
  bool needs_commit() const;
  void commit(std::function<void()> completion = nullptr);
  void synchronize();

  MTL::CommandBuffer* get_command_buffer() const {
    return buffer_.get();
  }

  // Keep an object (array data, a fence) alive until the command buffer that
  // is current now completes. Everything retained is released by the single
  // completion handler commit() installs, instead of one Metal completion
  // handler (an Objective-C block plus a hash set) per encoded primitive.
  // A primitive that commits mid-eval (e.g. the expert-tile route's retract
  // check) retains into the replacement buffer, which completes after the
  // committed one on this queue, so the lifetime is never shorter.
  void retain_until_completed(std::shared_ptr<void> object) {
    retained_.push_back(std::move(object));
  }

 private:
  MTL::ComputeCommandEncoder* get_command_encoder();

  Device& device_;
  bool exiting_{false};

  // Buffer that stores encoded commands.
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  size_t buffer_sizes_{0};

  // The residency set and how many of its sets this queue has attached.
  ResidencySets& residency_sets_;
  uint64_t sets_attached_{0};

  // The events hooked to current command buffer.
  std::vector<Event> wait_events_;
  std::vector<std::tuple<Event, uint64_t>> signal_events_;

  // Error from previous commited command buffer.
  Error error_;

  // Encoder for issuing GPU commands.
  // The members are used within a single ComputeCommandEncoder and will be
  // reset after calling end_encoding().
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  NS::SharedPtr<MTL::Fence> fence_;
  bool needs_barrier_{false};
  bool wait_all_pending_{false};
  bool concurrent_{false};
  std::vector<array> temporaries_;
  BufferSet prev_inputs_;
  BufferSet prev_outputs_;
  BufferSet next_inputs_;
  BufferSet next_outputs_;
  BufferSet concurrent_outputs_;
  BufferSet all_inputs_;
  BufferSet all_outputs_;

  // A map of prior command encoder outputs to their corresponding fence.
  std::unordered_map<const void*, NS::SharedPtr<MTL::Fence>> prev_ce_outputs_;
  std::mutex outputs_mtx_;

  // Objects kept alive until the current command buffer completes
  // (`retain_until_completed`); moved into its completion handler at commit.
  std::vector<std::shared_ptr<void>> retained_;
};

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_.get();
  };

  const std::string& get_architecture() const {
    return arch_;
  }
  int get_architecture_gen() const {
    return arch_gen_;
  }
  // The device defaults, or the calling thread's limits where set
  // (mlx_metal_set_thread_command_buffer_limits).
  std::tuple<int, int> get_max_ops_mb_per_buffer() const;

  MTL::Library* get_library(
      const std::string& name,
      const std::string& path = "");

  MTL::Library* get_library(
      const std::string& name,
      const CompileOptions& compile_options,
      const std::function<std::string(void)>& builder);

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder) {
    return get_library(name, {}, builder);
  }

  void clear_library(const std::string& name);

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  ResidencySets& residency_sets() {
    return residency_sets_;
  }

  bool gemma4_expert_qmm_requested() const {
    return gemma4_expert_qmm_requested_;
  }

  // MLX_GATHER_QMM_EXPERT_SLICES=trust: skip the descriptor-retract
  // readback in the expert-tile route (no mid-eval stream drain). The
  // caller asserts sorted indices are machine-guaranteed; a violation
  // yields undefined tile output instead of the legacy fallback.
  bool gemma4_expert_qmm_trust_sorted() const {
    return gemma4_expert_qmm_trust_sorted_;
  }

  bool gemma4_expert_qmm_aot_available() const {
    return gemma4_expert_qmm_aot_available_;
  }
  bool gemma4_expert_qmm_diagnostics_armed() const {
    return gemma4_expert_qmm_counters_.armed();
  }

  // Call only inside a route boundary guarded by
  // gemma4_expert_qmm_diagnostics_armed().
  void record_armed_gemma4_expert_qmm(Gemma4ExpertQMMRoute route) {
    gemma4_expert_qmm_counters_.record(route);
  }

  Gemma4ExpertQMMCounterSnapshot gemma4_expert_qmm_counter_snapshot() const {
    return gemma4_expert_qmm_counters_.snapshot();
  }
  Gemma4ExpertQMMCounterSnapshot
  gemma4_expert_qmm_counter_snapshot_and_disarm() {
    return gemma4_expert_qmm_counters_.snapshot_and_disarm();
  }

  void reset_gemma4_expert_qmm_counters() {
    gemma4_expert_qmm_counters_.reset();
  }
  void clear_and_arm_gemma4_expert_qmm_counters() {
    gemma4_expert_qmm_counters_.clear_and_arm();
  }

 private:
  NS::SharedPtr<MTL::Library> build_library_(
      const std::string& source_string,
      const CompileOptions& compile_options = {});

  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      MTL::Library* mtl_lib);
  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  NS::SharedPtr<MTL::LinkedFunctions> get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);
  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  NS::SharedPtr<MTL::Device> device_;
  ResidencySets residency_sets_;

  std::shared_mutex kernel_mtx_;
  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::Library>> library_map_;
  NS::SharedPtr<MTL::Library> default_library_;
  bool gemma4_expert_qmm_requested_{false};
  bool gemma4_expert_qmm_trust_sorted_{false};
  bool gemma4_expert_qmm_aot_available_{false};
  Gemma4ExpertQMMCounters gemma4_expert_qmm_counters_;
  std::unordered_map<
      MTL::Library*,
      std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>
      library_kernels_;
  std::string arch_;
  int arch_gen_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

MLX_API Device& device(mlx::core::Device);
MLX_API CommandEncoder& get_command_encoder(Stream s);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();
NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool();

bool is_nax_available();

} // namespace mlx::core::metal
