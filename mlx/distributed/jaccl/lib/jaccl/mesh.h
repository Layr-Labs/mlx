// Copyright © 2026 Apple Inc.

#pragma once

#include "jaccl/group.h"
#include "jaccl/mesh_impl.h"
#include "jaccl/rdma.h"

namespace jaccl {

/**
 * The JACCL communication group for a fully connected mesh. We expect one
 * connection per peer and it should be the lowest latency communication group
 * for small to medium size messages.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class MeshGroup : public Group {
 public:
  MeshGroup(
      int rank,
      const std::vector<std::string>& device_names,
      SideChannel sc);

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void begin_startup_wait_budget(int64_t timeout_ms) override {
    mesh_.begin_startup_wait_budget(timeout_ms);
  }

  void end_startup_wait_budget() override {
    mesh_.end_startup_wait_budget();
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_gather(const void* input, void* output, size_t n_bytes) override;

  void sum_scatter(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void send(const void* input, size_t n_bytes, int dst) override;
  void recv(void* output, size_t n_bytes, int src) override;

  void barrier() override;

  bool supports_exchange() override {
    return size_ == 2;
  }

  void exchange(
      const char* input,
      char* output,
      size_t n_bytes,
      const std::function<bool(size_t)>& ready,
      const std::function<void(size_t)>& arrived) override {
    mesh_.exchange(input, output, n_bytes, ready, arrived);
  }

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  template <typename T, typename ReduceOp>
  void reduce_scatter(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state and all buffers
   * should have been allocated.
   */
  void initialize();

  /**
   * Allocate all the buffers that we will use in the communication group.
   */
  void allocate_buffers();

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
  std::vector<SharedBuffer> scatter_buffers_;

  MeshImpl mesh_;
};

} // namespace jaccl
