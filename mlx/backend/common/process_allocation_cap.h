// Copyright © 2026 Eigen Labs.
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace mlx::core::allocator {

struct ProcessAllocationCapSnapshot {
  bool installed{false};
  size_t limit_bytes{0};
  size_t heap_bytes{0};
  size_t live_bytes{0};
  size_t cached_bytes{0};
  size_t pending_bytes{0};
  size_t unreclaimed_bytes{0};
  size_t charged_bytes{0};
  size_t peak_bytes{0};
};

// Reference release alone cannot refund backing credit.
class ProcessAllocationCap {
 public:
  void install(size_t limit, size_t heap) {
    std::lock_guard lock(mutex_);
    if (state_.installed || !limit || heap > limit) {
      throw std::invalid_argument("allocation cap install refused");
    }
    state_.installed = true;
    state_.limit_bytes = limit;
    state_.heap_bytes = heap;
    state_.charged_bytes = state_.peak_bytes = heap;
  }

  void reserve(size_t bytes) {
    std::lock_guard lock(mutex_);
    if (!state_.installed ||
        bytes > state_.limit_bytes - state_.charged_bytes) {
      throw std::runtime_error("native allocation cap exceeded");
    }
    state_.pending_bytes += bytes;
    state_.charged_bytes += bytes;
    if (state_.charged_bytes > state_.peak_bytes) {
      state_.peak_bytes = state_.charged_bytes;
    }
  }

  // Only a reservation for which no backing was acquired can be rolled back.
  void failed_before_acquisition(size_t bytes) {
    std::lock_guard lock(mutex_);
    require(state_.pending_bytes, bytes);
    state_.pending_bytes -= bytes;
    state_.charged_bytes -= bytes;
  }
  void acquired(size_t bytes) {
    std::lock_guard lock(mutex_);
    transfer(state_.pending_bytes, state_.live_bytes, bytes);
  }
  void cache(size_t bytes) {
    std::lock_guard lock(mutex_);
    transfer(state_.live_bytes, state_.cached_bytes, bytes);
  }
  void reuse(size_t bytes) {
    std::lock_guard lock(mutex_);
    transfer(state_.cached_bytes, state_.live_bytes, bytes);
  }
  void relinquish(size_t bytes, bool cached = false) {
    std::lock_guard lock(mutex_);
    transfer(
        cached ? state_.cached_bytes : state_.live_bytes,
        state_.unreclaimed_bytes,
        bytes);
  }
  void destroyed(size_t bytes) {
    std::lock_guard lock(mutex_);
    require(state_.unreclaimed_bytes, bytes);
    state_.unreclaimed_bytes -= bytes;
    state_.charged_bytes -= bytes;
  }
  ProcessAllocationCapSnapshot snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

 private:
  static void require(size_t available, size_t bytes) {
    if (bytes > available) {
      throw std::logic_error("allocation cap ledger mismatch");
    }
  }
  static void transfer(size_t& from, size_t& to, size_t bytes) {
    require(from, bytes);
    if (bytes > std::numeric_limits<size_t>::max() - to) {
      throw std::overflow_error("allocation cap ledger overflow");
    }
    from -= bytes;
    to += bytes;
  }
  mutable std::mutex mutex_;
  ProcessAllocationCapSnapshot state_;
};

// Shared by allocator bookkeeping and the Metal deallocator block. The callback
// never takes the allocator mutex. Release must finish before credit is
// returned.
class ProcessBackingLease {
 public:
  using Release = bool (*)(void*, size_t);
  ProcessBackingLease(std::shared_ptr<ProcessAllocationCap> cap, size_t bytes)
      : cap_(std::move(cap)), bytes_(bytes) {
    cap_->reserve(bytes_);
  }
  ~ProcessBackingLease() {
    finish();
  }
  void attach(void* pointer, Release release) noexcept {
    std::lock_guard lock(mutex_);
    pointer_ = pointer;
    release_ = release;
  }
  void acquired() {
    std::lock_guard lock(mutex_);
    if (state_ != State::pending) {
      throw std::logic_error("backing already retired");
    }
    cap_->acquired(bytes_);
    state_ = State::live;
  }
  void cache() {
    std::lock_guard lock(mutex_);
    if (state_ != State::live) {
      throw std::logic_error("backing cache mismatch");
    }
    cap_->cache(bytes_);
    state_ = State::cached;
  }
  void reuse() {
    std::lock_guard lock(mutex_);
    if (state_ != State::cached) {
      throw std::logic_error("backing reuse mismatch");
    }
    cap_->reuse(bytes_);
    state_ = State::live;
  }
  void retire() {
    std::lock_guard lock(mutex_);
    if (state_ != State::live && state_ != State::cached) {
      throw std::logic_error("backing retirement mismatch");
    }
    cap_->relinquish(bytes_, state_ == State::cached);
    state_ = State::retired;
  }
  void finish() noexcept {
    std::lock_guard lock(mutex_);
    if (state_ == State::finished) {
      return;
    }
    // Foreign wrappers have no release function: their caller retains
    // ownership.
    const bool released = !release_ || release_(pointer_, bytes_);
    if (state_ == State::pending && released) {
      cap_->failed_before_acquisition(bytes_);
    } else {
      if (state_ == State::pending) {
        cap_->acquired(bytes_);
      }
      if (state_ != State::retired) {
        cap_->relinquish(bytes_, state_ == State::cached);
      }
      if (released) {
        cap_->destroyed(bytes_);
      }
    }
    // A failed backing release is permanently charged; it cannot be retried
    // after ownership has been relinquished to a deallocator callback.
    state_ = State::finished;
  }

 private:
  enum class State { pending, live, cached, retired, finished };
  std::shared_ptr<ProcessAllocationCap> cap_;
  size_t bytes_;
  void* pointer_{nullptr};
  Release release_{nullptr};
  State state_{State::pending};
  std::mutex mutex_;
};

} // namespace mlx::core::allocator
