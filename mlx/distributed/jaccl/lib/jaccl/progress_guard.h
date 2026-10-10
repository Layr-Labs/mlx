// Copyright © 2026 Apple Inc.

#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace jaccl {

constexpr int64_t DEFAULT_PROGRESS_TIMEOUT_MS = 30000;
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

// What a call does after it failed and closed its group.
enum class FailureAction {
  // Release the group, run the memory release hook, exit with status 75.
  TeardownExit,
  // Throw to the caller; the process continues with the group closed.
  Throw,
};

// EX_TEMPFAIL: the failure is temporary and the process can be restarted.
constexpr int TEARDOWN_EXIT_STATUS = 75;
// The memory release hook gets this long, then the process exits anyway.
constexpr int64_t TEARDOWN_RELEASE_LIMIT_MS = 10000;

/**
 * The action after a failed call. It is read once from JACCL_TIMEOUT_ACTION /
 * MLX_JACCL_TIMEOUT_ACTION: "teardown-exit" (the default) or "throw". Any
 * other value keeps the default.
 */
inline FailureAction failure_action() {
  static const FailureAction action = []() {
    const char* value = std::getenv("JACCL_TIMEOUT_ACTION");
    if (value == nullptr) {
      value = std::getenv("MLX_JACCL_TIMEOUT_ACTION");
    }
    if (value != nullptr && std::strcmp(value, "throw") == 0) {
      return FailureAction::Throw;
    }
    return FailureAction::TeardownExit;
  }();
  return action;
}

// Releases the memory of the embedding library before a teardown exit, for
// example the wired GPU buffers that would otherwise stay wired after the
// process is gone. It must not throw.
using MemoryRelease = void (*)();

inline std::atomic<MemoryRelease>& memory_release_hook() {
  static std::atomic<MemoryRelease> hook{nullptr};
  return hook;
}

// Register the hook that teardown_exit() runs. The last registration wins.
inline void set_memory_release(MemoryRelease release) {
  memory_release_hook().store(release);
}

/**
 * Leave the process after a failed call when the action is teardown-exit.
 * The caller has already destroyed the queue pairs and completion queues of
 * the group and deregistered its buffers. This prints the reason, runs the
 * memory release hook (at most TEARDOWN_RELEASE_LIMIT_MS), then exits with
 * TEARDOWN_EXIT_STATUS without running atexit handlers or static destructors.
 * Only the first caller does so; a thread that fails at the same time waits
 * for that exit.
 */
[[noreturn]] inline void teardown_exit(const char* reason) noexcept {
  static std::atomic<bool> exiting{false};
  if (exiting.exchange(true)) {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  std::fprintf(
      stderr,
      "%s\n[jaccl] The group is released; releasing memory, then "
      "exiting with status %d (JACCL_TIMEOUT_ACTION=teardown-exit).\n",
      reason,
      TEARDOWN_EXIT_STATUS);
  std::fflush(stderr);
  MemoryRelease release = memory_release_hook().load();
  if (release != nullptr) {
    try {
      std::thread([] {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(TEARDOWN_RELEASE_LIMIT_MS));
        std::fprintf(
            stderr,
            "[jaccl] The memory release did not finish in %lld ms; exiting.\n",
            static_cast<long long>(TEARDOWN_RELEASE_LIMIT_MS));
        std::fflush(stderr);
        std::_Exit(TEARDOWN_EXIT_STATUS);
      }).detach();
    } catch (...) {
      // Without the time limit the release still runs.
    }
    try {
      release();
    } catch (...) {
      // The hook must not throw; the exit goes ahead.
    }
  }
  std::fprintf(
      stderr,
      "[jaccl] %s; exiting with status %d.\n",
      release != nullptr ? "Memory released" : "No memory release registered",
      TEARDOWN_EXIT_STATUS);
  std::fflush(stderr);
  std::_Exit(TEARDOWN_EXIT_STATUS);
}

// The text of an exception, for teardown_exit().
inline std::string describe(const std::exception_ptr& error) {
  try {
    std::rethrow_exception(error);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "[jaccl] A call failed with an exception that is not std::exception.";
  }
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
 * before it throws, which also stops the other wires of the same call. The
 * owner of the group closes it and then, unless failure_action() is Throw,
 * calls teardown_exit().
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

 protected:
  // The error that check() threw, or empty.
  const std::string& failure() const {
    return failure_;
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
    failure_ = msg.str();
    throw std::runtime_error(failure_);
  }

  const char* op_;
  std::atomic<bool>& failed_;
  int64_t timeout_ms_;
  int64_t idle_polls_{0};
  std::chrono::steady_clock::time_point idle_since_;
  std::string failure_;
};

} // namespace jaccl
