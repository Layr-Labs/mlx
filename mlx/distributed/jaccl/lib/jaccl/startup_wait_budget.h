// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace jaccl {

constexpr int64_t MAX_STARTUP_WAIT_BUDGET_MS = 900000;

// Clock injection is local to the type; production uses steady_clock.
template <typename Clock = std::chrono::steady_clock>
class BasicStartupWaitBudget {
 public:
  using TimePoint = typename Clock::time_point;

  void begin(int64_t timeout_ms, const std::atomic<bool>& failed) {
    if (timeout_ms <= 0 || timeout_ms > MAX_STARTUP_WAIT_BUDGET_MS) {
      throw std::invalid_argument(
          "[jaccl] Startup budget must be 1..900000 ms.");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    require_idle_open(failed);
    if (state_ != State::Unused) {
      throw std::logic_error("[jaccl] Startup budget can only begin once.");
    }
    using Duration = typename Clock::duration;
    static_assert(Clock::is_steady);
    static_assert(
        std::chrono::duration_cast<std::chrono::milliseconds>(Duration::max())
            .count() >= MAX_STARTUP_WAIT_BUDGET_MS);
    const auto delta = std::chrono::duration_cast<Duration>(
        std::chrono::milliseconds(timeout_ms));
    const auto now = Clock::now();
    if (now.time_since_epoch() > Duration::max() - delta) {
      throw std::overflow_error(
          "[jaccl] Startup deadline overflows the clock.");
    }
    deadline_ = now + delta;
    state_ = State::Active;
  }

  void end(const std::atomic<bool>& failed) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_idle_open(failed);
    if (state_ != State::Active) {
      throw std::logic_error("[jaccl] No active startup budget to end.");
    }
    if (Clock::now() >= *deadline_) {
      throw std::logic_error("[jaccl] Cannot end an expired startup budget.");
    }
    state_ = State::Ended;
  }

  // Each operation holds a lease until all its wires and teardown finish.
  class Call {
   public:
    explicit Call(BasicStartupWaitBudget& owner) : owner_(owner) {
      std::lock_guard<std::mutex> lock(owner_.mutex_);
      if (owner_.state_ == State::Active) {
        deadline_ = owner_.deadline_;
      }
      ++owner_.calls_;
    }

    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;

    ~Call() {
      std::lock_guard<std::mutex> lock(owner_.mutex_);
      --owner_.calls_;
    }

    const std::optional<TimePoint>& deadline() const {
      return deadline_;
    }

   private:
    BasicStartupWaitBudget& owner_;
    std::optional<TimePoint> deadline_;
  };

 private:
  enum class State { Unused, Active, Ended };

  void require_idle_open(const std::atomic<bool>& failed) const {
    if (failed.load()) {
      throw std::runtime_error(
          "[jaccl] The group is closed after an earlier failure and cannot be "
          "used again.");
    }
    if (calls_ != 0) {
      throw std::logic_error(
          "[jaccl] Startup transition requires an idle group.");
    }
  }

  std::mutex mutex_;
  State state_{State::Unused};
  std::size_t calls_{0};
  std::optional<TimePoint> deadline_;
};

using StartupWaitBudget = BasicStartupWaitBudget<>;

} // namespace jaccl
