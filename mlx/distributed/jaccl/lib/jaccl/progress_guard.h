// Copyright © 2026 Apple Inc.

#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace jaccl {

constexpr int64_t DEFAULT_PROGRESS_TIMEOUT_MS = 120000;
// The clock is read once per this number of polls without a completion.
constexpr int64_t PROGRESS_SAMPLE_POLLS = 1024;

/**
 * The longest time in milliseconds that a call waits for a completion. It is
 * read once from JACCL_PROGRESS_TIMEOUT_MS / MLX_JACCL_PROGRESS_TIMEOUT_MS.
 * Zero or a negative value removes the limit.
 */
inline int64_t progress_timeout_ms() {
  static const int64_t timeout = []() -> int64_t {
    const char* value = std::getenv("JACCL_PROGRESS_TIMEOUT_MS");
    if (value == nullptr) {
      value = std::getenv("MLX_JACCL_PROGRESS_TIMEOUT_MS");
    }
    if (value == nullptr) {
      return DEFAULT_PROGRESS_TIMEOUT_MS;
    }
    // Keep the default when the value is not a number.
    char* end = nullptr;
    errno = 0;
    long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE) {
      return DEFAULT_PROGRESS_TIMEOUT_MS;
    }
    return parsed;
  }();
  return timeout;
}

// Thrown in a wire when a different wire of the same call failed first.
struct WireStopped : std::runtime_error {
  WireStopped()
      : std::runtime_error(
            "[jaccl] Stopped because a different wire of the call failed.") {}
};

// Refuse a group that an earlier failure closed.
inline void require_open(const std::atomic<bool>& failed) {
  if (failed.load()) {
    throw std::runtime_error(
        "[jaccl] The group is closed after an earlier failure and cannot be "
        "used again.");
  }
}

/**
 * Stops a polling loop that makes no progress. Call check() after each poll.
 *
 * check() throws when a completion reports a failure, when the poll fails or
 * when no completion arrived for longer than the timeout. It sets `failed`
 * before it throws, which also stops the other wires of the same call.
 */
class ProgressGuard {
 public:
  ProgressGuard(const char* op, std::atomic<bool>& failed)
      : op_(op), failed_(failed), timeout_ms_(progress_timeout_ms()) {}

  void check(const ibv_wc* wc, int n) {
    if (n > 0) {
      idle_polls_ = 0;
      for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::ostringstream msg;
          msg << "a work completion failed with status " << wc[i].status
              << " (vendor error " << wc[i].vendor_err << ", id 0x" << std::hex
              << wc[i].wr_id << ")";
          fail(msg.str());
        }
      }
    } else if (n < 0) {
      fail("the completion queue could not be polled");
    } else if (++idle_polls_ % PROGRESS_SAMPLE_POLLS == 0) {
      check_idle();
    }
  }

 private:
  void check_idle() {
    if (failed_.load()) {
      throw WireStopped();
    }
    if (timeout_ms_ <= 0) {
      return;
    }
    auto now = std::chrono::steady_clock::now();
    // The first sample after a completion starts the measurement.
    if (idle_polls_ == PROGRESS_SAMPLE_POLLS) {
      idle_since_ = now;
      return;
    }
    int64_t idle_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - idle_since_)
            .count();
    if (idle_ms > timeout_ms_) {
      std::ostringstream msg;
      msg << "no completion for " << idle_ms
          << " ms (JACCL_PROGRESS_TIMEOUT_MS is " << timeout_ms_ << ")";
      fail(msg.str());
    }
  }

  [[noreturn]] void fail(const std::string& reason) {
    failed_.store(true);
    std::ostringstream msg;
    msg << "[jaccl] " << op_ << ": " << reason
        << ". The group is closed and cannot be used again.";
    throw std::runtime_error(msg.str());
  }

  const char* op_;
  std::atomic<bool>& failed_;
  int64_t timeout_ms_;
  int64_t idle_polls_{0};
  std::chrono::steady_clock::time_point idle_since_;
};

} // namespace jaccl
