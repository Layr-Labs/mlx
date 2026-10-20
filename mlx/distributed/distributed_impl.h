// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

/**
 * Abstract base class of a distributed group implementation.
 */
class GroupImpl {
 public:
  virtual ~GroupImpl() {}

  // Choose the stream this communication group can operate on
  virtual Stream communication_stream(StreamOrDevice s = {}) = 0;

  // Group operations
  virtual int rank() = 0;
  virtual int size() = 0;
  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) = 0;

  virtual void begin_startup_wait_budget(int64_t) {
    throw std::runtime_error(
        "[distributed] Startup wait budgets require a JACCL group.");
  }
  virtual void end_startup_wait_budget() {
    throw std::runtime_error(
        "[distributed] Startup wait budgets require a JACCL group.");
  }

  // Actual communication operations
  virtual void all_sum(const array& input, array& output, Stream stream) = 0;
  virtual void all_gather(const array& input, array& output, Stream stream) = 0;
  virtual void send(const array& input, int dst, Stream stream) = 0;
  virtual void recv(array& out, int src, Stream stream) = 0;
  virtual void all_max(const array& input, array& output, Stream stream) = 0;
  virtual void all_min(const array& input, array& output, Stream stream) = 0;
  virtual void
  sum_scatter(const array& input, array& output, Stream stream) = 0;

  // Two ranks, on the calling thread (a CPU-stream task): send n bytes while
  // the peer's n bytes land in recv. ready(end) tells if send[0, end) is
  // final; it is polled and must not block. arrived(end) runs when
  // recv[0, end) holds the peer's bytes. Neither callback may throw.
  virtual bool supports_exchange() {
    return false;
  }
  virtual void exchange(
      const char* /* send */,
      char* /* recv */,
      size_t /* n */,
      const std::function<bool(size_t)>& /* ready */,
      const std::function<void(size_t)>& /* arrived */) {
    throw std::runtime_error("[distributed] This group cannot exchange.");
  }
};

/* Define the MLX stream that the communication should happen in. */
Stream communication_stream(Group group, StreamOrDevice s = {});

/* Perform an all reduce sum operation */
void all_sum(Group group, const array& input, array& output, Stream stream);

/* Perform an all gather operation */
void all_gather(Group group, const array& input, array& output, Stream stream);

/** Send an array to the dst rank */
void send(Group group, const array& input, int dst, Stream stream);

/** Recv an array from the src rank */
void recv(Group group, array& out, int src, Stream stream);

/** Max reduction */
void all_max(Group group, const array& input, array& output, Stream stream);

/** Min reduction */
void all_min(Group group, const array& input, array& output, Stream stream);

/** Reduce scatter with average operation */
void sum_scatter(Group group, const array& input, array& output, Stream stream);

} // namespace mlx::core::distributed::detail
