// Copyright © 2026 Darkbloom.
//
// The streamed sealed sum. Worker threads seal the pieces in order; the
// exchange sends each frame when its pieces are sealed; each peer piece is
// opened and added when its bytes land. Only the head and the tail of the
// transfer wait for crypto.

#pragma once

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <pthread/qos.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>

#include "mlx/distributed/sealed.h"

namespace mlx::core::distributed::detail {

using ReadyFn = std::function<bool(size_t)>;
using ArrivedFn = std::function<void(size_t)>;
// exchange(send, recv, bytes, ready, arrived): see GroupImpl::exchange.
using ExchangeFn = std::function<
    void(const char*, char*, size_t, const ReadyFn&, const ArrivedFn&)>;

// The streamed frame of an n-byte partial.
struct PieceLayout {
  size_t bytes;
  size_t piece;
  size_t count;
  PieceLayout(size_t n, size_t piece_bytes)
      : bytes(n),
        piece(piece_bytes),
        count(std::max<size_t>(1, (n + piece_bytes - 1) / piece_bytes)) {}
  size_t plain(size_t k) const {
    return k * piece;
  }
  size_t length(size_t k) const {
    return std::min(piece, bytes - k * piece);
  }
  size_t sealed(size_t k) const {
    return k * (piece + kSealTagBytes);
  }
  size_t sealed_end(size_t k) const {
    return sealed(k) + length(k) + kSealTagBytes;
  }
  size_t frame() const {
    return bytes + count * kSealTagBytes;
  }
};

// `out` holds the peer's addend. It becomes rank 0's addend + rank 1's, both
// widened to float and rounded once, as the gathered path adds them.
template <typename T>
void add_peer_in_rank_order(
    int rank,
    const T* __restrict mine,
    T* __restrict out,
    size_t n) {
  if (rank == 0) {
    for (size_t i = 0; i < n; ++i) {
      out[i] = static_cast<T>(
          static_cast<float>(mine[i]) + static_cast<float>(out[i]));
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      out[i] = static_cast<T>(
          static_cast<float>(out[i]) + static_cast<float>(mine[i]));
    }
  }
}

using AddPieceFn = void (*)(int rank, const char* mine, char* out, size_t len);

template <typename T>
void add_piece(int rank, const char* mine, char* out, size_t len) {
  add_peer_in_rank_order(
      rank,
      reinterpret_cast<const T*>(mine),
      reinterpret_cast<T*>(out),
      len / sizeof(T));
}

// MLX_SEALED_QOS=0 keeps the caller's QoS (as the gathered path does).
inline bool raise_qos() {
  static const bool on = [] {
    const char* value = std::getenv("MLX_SEALED_QOS");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  return on;
}

// Threads that seal at the same time (MLX_SEALED_SEAL_LANES, 1..64). Eight
// stay ahead of the wire and leave cores to the GPU driver and the opens.
inline size_t seal_lanes() {
  static const size_t lanes = [] {
    const char* value = std::getenv("MLX_SEALED_SEAL_LANES");
    long parsed = value ? std::strtol(value, nullptr, 10) : 0;
    return parsed >= 1 && parsed <= 64 ? size_t(parsed) : size_t(8);
  }();
  return lanes;
}

// One streamed sealed exchange, run on the calling thread (a CPU-stream
// task): `send` is sealed and sent piece by piece; each peer piece is opened
// into `out` and, with `add`, `addend` (this rank's own values for the same
// elements) is added to it in rank order. The all-sum sends and adds its
// whole partial; the reduce-scatter sends the peer's half and adds its own;
// the all-gather sends its own half and only opens.
class StreamedSum {
 public:
  StreamedSum(
      const SealCodec& codec,
      const PieceLayout& layout,
      uint64_t counter,
      int rank,
      const char* send,
      const char* addend,
      char* out,
      char* sealed,
      char* received,
      AddPieceFn add)
      : codec_(codec),
        layout_(layout),
        counter_(counter),
        rank_(rank),
        mine_(send),
        addend_(addend),
        out_(out),
        sealed_(sealed),
        received_(received),
        add_(add),
        ready_(std::make_unique<std::atomic<bool>[]>(layout.count)) {}

  // False when a piece did not seal or a peer piece did not open. No worker
  // runs after this returns or throws.
  bool run(const ExchangeFn& exchange) {
    if (layout_.count == 1) {
      // One piece (decode): no worker hand-off.
      seal(0);
      exchange(
          sealed_,
          received_,
          layout_.frame(),
          [](size_t) { return true; },
          [this](size_t end) {
            if (end == layout_.frame()) {
              open_and_add(0);
            }
          });
      return !failed_.load();
    }
    QosOverride boost;
    Workers workers;
    StreamedSum* self = this;
    size_t lanes = std::min(layout_.count, seal_lanes());
    for (size_t i = 0; i < lanes; ++i) {
      workers.async(^{
        self->seal_in_order();
      });
    }
    size_t sealed_prefix = 0;
    size_t dispatched = 0;
    exchange(
        sealed_,
        received_,
        layout_.frame(),
        [&](size_t end) {
          while (sealed_prefix < layout_.count &&
                 ready_[sealed_prefix].load(std::memory_order_acquire)) {
            ++sealed_prefix;
          }
          return sealed_prefix == layout_.count ||
              layout_.sealed(sealed_prefix) >= end;
        },
        [&](size_t end) {
          for (; dispatched < layout_.count &&
               layout_.sealed_end(dispatched) <= end;
               ++dispatched) {
            size_t k = dispatched;
            workers.async(^{
              self->open_and_add(k);
            });
          }
        });
    workers.wait();
    return !failed_.load();
  }

 private:
  // Waits for every block before the frames and `out` can be released.
  class Workers {
   public:
    Workers()
        : group_(dispatch_group_create()),
          queue_(dispatch_get_global_queue(
              raise_qos() ? QOS_CLASS_USER_INTERACTIVE : qos_class_self(),
              0)) {}
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;
    ~Workers() {
      wait();
      dispatch_release(group_);
    }
    void async(dispatch_block_t block) {
      dispatch_group_async(group_, queue_, block);
    }
    void wait() {
      dispatch_group_wait(group_, DISPATCH_TIME_FOREVER);
    }

   private:
    dispatch_group_t group_;
    dispatch_queue_t queue_;
  };

  // The polling thread runs at the workers' QoS while they run.
  class QosOverride {
   public:
    QosOverride()
        : token_(
              raise_qos() ? pthread_override_qos_class_start_np(
                                pthread_self(),
                                QOS_CLASS_USER_INTERACTIVE,
                                0)
                          : nullptr) {}
    QosOverride(const QosOverride&) = delete;
    QosOverride& operator=(const QosOverride&) = delete;
    ~QosOverride() {
      if (token_ != nullptr) {
        pthread_override_qos_class_end_np(token_);
      }
    }

   private:
    pthread_override_t token_;
  };

  void seal_in_order() {
    for (size_t k; (k = next_seal_.fetch_add(1)) < layout_.count;) {
      seal(k);
    }
  }

  void seal(size_t k) {
    char* dst = sealed_ + layout_.sealed(k);
    size_t len = layout_.length(k);
    if (codec_.seal_piece(
            codec_.ctx,
            mine_ + layout_.plain(k),
            len,
            dst,
            counter_,
            static_cast<uint32_t>(k),
            static_cast<uint32_t>(layout_.count),
            layout_.bytes) != 0) {
      // Never send stale bytes. The peer refuses the zero piece.
      std::memset(dst, 0, len + kSealTagBytes);
      failed_.store(true);
    }
    ready_[k].store(true, std::memory_order_release);
  }

  void open_and_add(size_t k) {
    char* dst = out_ + layout_.plain(k);
    size_t len = layout_.length(k);
    if (codec_.open_piece(
            codec_.ctx,
            received_ + layout_.sealed(k),
            len,
            dst,
            counter_,
            static_cast<uint32_t>(k),
            static_cast<uint32_t>(layout_.count),
            layout_.bytes,
            1 - rank_) != 0) {
      failed_.store(true);
      return;
    }
    if (add_ != nullptr) {
      add_(rank_, addend_ + layout_.plain(k), dst, len);
    }
  }

  const SealCodec codec_;
  const PieceLayout layout_;
  const uint64_t counter_;
  const int rank_;
  const char* const mine_;
  const char* const addend_;
  char* const out_;
  char* const sealed_;
  char* const received_;
  const AddPieceFn add_;
  std::unique_ptr<std::atomic<bool>[]> ready_;
  std::atomic<size_t> next_seal_{0};
  std::atomic<bool> failed_{false};
};

} // namespace mlx::core::distributed::detail
