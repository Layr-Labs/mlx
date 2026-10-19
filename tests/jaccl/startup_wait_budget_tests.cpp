// Copyright © 2026 Apple Inc.
// Standalone CPU policy tests: use the real policy and guard with a fake clock.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct FakeClock {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<FakeClock>;
  static constexpr bool is_steady = true;
  static inline std::atomic<int64_t> ticks{0};
  static time_point now() {
    return time_point(duration(ticks.load()));
  }
  static void ms(int64_t n) {
    ticks.store(n * 1000000);
  }
};

#include "jaccl/progress_guard.h"

using Policy = jaccl::BasicStartupWaitBudget<FakeClock>;
using Guard = jaccl::BasicProgressGuard<FakeClock>;
using namespace std::chrono_literals;

void require(bool ok) {
  if (!ok) {
    throw std::runtime_error("assertion failed");
  }
}
template <typename E = std::exception, typename Fn>
void refuses(Fn fn, const std::string& message = "") {
  try {
    fn();
  } catch (const E& e) {
    require(
        message.empty() ||
        std::string(e.what()).find(message) != std::string::npos);
    return;
  }
  throw std::runtime_error("expected refusal");
}
void sample(Guard& g) {
  for (int i = 0; i < jaccl::PROGRESS_SAMPLE_POLLS; ++i) {
    g.check(nullptr, 0);
  }
}
ibv_wc success{};

void default_idle() {
  std::atomic<bool> failed{false};
  Guard g("default", failed);
  sample(g);
  FakeClock::ms(30000);
  sample(g);
  require(!failed);
  FakeClock::ms(30001);
  refuses([&] { sample(g); }, "no completion for 30001");
  require(failed);
}
void default_progress_resets_idle() {
  std::atomic<bool> failed{false};
  Policy p;
  Guard g("default group", failed, &p);
  sample(g);
  FakeClock::ms(29000);
  g.check(&success, 1);
  sample(g);
  FakeClock::ms(59000);
  sample(g);
  require(!failed);
  FakeClock::ms(59001);
  refuses([&] { sample(g); });
}
void startup_deadline() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(120000, failed);
  Guard g("startup", failed, &p);
  sample(g);
  FakeClock::ms(30001);
  sample(g);
  FakeClock::ms(119999);
  sample(g);
  require(!failed);
  FakeClock::ms(120000);
  refuses([&] { sample(g); }, "absolute startup");
  require(failed);
}
void progress_cannot_renew() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(60000, failed);
  Guard g("progress", failed, &p);
  for (int i = 1; i < 60; ++i) {
    FakeClock::ms(i * 1000);
    g.check(&success, 1);
  }
  FakeClock::ms(60000);
  refuses([&] { g.check(&success, 1); }, "absolute startup");
  require(failed);
}
void deadline_shared_across_calls() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(60000, failed);
  {
    Guard first("first", failed, &p);
    FakeClock::ms(40000);
    first.check(&success, 1);
  }
  Guard second("second", failed, &p);
  FakeClock::ms(60000);
  refuses([&] { second.check(&success, 1); });
}
void end_restores_idle() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(120000, failed);
  {
    Guard startup("startup", failed, &p);
    FakeClock::ms(40000);
    startup.check(&success, 1);
  }
  p.end(failed);
  Guard serving("serving", failed, &p);
  sample(serving);
  FakeClock::ms(70000);
  sample(serving);
  require(!failed);
  FakeClock::ms(70001);
  refuses([&] { sample(serving); }, "no completion");
}
void invalid_does_not_consume() {
  std::atomic<bool> failed{false};
  Policy p;
  for (int64_t n :
       {int64_t(-1),
        int64_t(0),
        int64_t(900001),
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min()}) {
    refuses<std::invalid_argument>([&] { p.begin(n, failed); });
  }
  p.begin(1, failed);
  Guard g("minimum", failed, &p);
  FakeClock::ms(1);
  refuses([&] { g.check(&success, 1); });
}
void maximum_budget() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(900000, failed);
  Guard g("maximum", failed, &p);
  FakeClock::ms(899999);
  g.check(&success, 1);
  FakeClock::ms(900000);
  refuses([&] { g.check(&success, 1); });
}
void checked_overflow() {
  std::atomic<bool> failed{false};
  Policy p;
  FakeClock::ticks.store(std::numeric_limits<int64_t>::max() - 999999);
  refuses<std::overflow_error>([&] { p.begin(1, failed); });
  FakeClock::ms(0);
  p.begin(1, failed);
  p.end(failed);
}
void exact_clock_limit() {
  std::atomic<bool> failed{false};
  Policy p;
  FakeClock::ticks.store(std::numeric_limits<int64_t>::max() - 1000000);
  p.begin(1, failed);
  Guard g("clock limit", failed, &p);
  FakeClock::ticks.store(std::numeric_limits<int64_t>::max());
  refuses([&] { g.check(&success, 1); });
}
void negative_clock_epoch() {
  std::atomic<bool> failed{false};
  Policy p;
  FakeClock::ticks.store(std::numeric_limits<int64_t>::min());
  p.begin(900000, failed);
  p.end(failed);
}
void one_shot() {
  std::atomic<bool> failed{false};
  Policy p;
  refuses<std::logic_error>([&] { p.end(failed); });
  p.begin(1000, failed);
  refuses<std::logic_error>([&] { p.begin(900000, failed); });
  p.end(failed);
  refuses<std::logic_error>([&] { p.end(failed); });
  refuses<std::logic_error>([&] { p.begin(1000, failed); });
}
void expired_end_refuses() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(1000, failed);
  FakeClock::ms(1000);
  refuses<std::logic_error>([&] { p.end(failed); }, "expired");
  refuses<std::logic_error>([&] { p.begin(1000, failed); });
  Guard g("still expired", failed, &p);
  refuses([&] { g.check(&success, 1); });
  require(failed);
}
void closed_group_stays_closed() {
  std::atomic<bool> failed{true};
  Policy p;
  refuses([&] { p.begin(1000, failed); });
  require(failed);
  std::atomic<bool> another_failed{false};
  Policy q;
  q.begin(1000, another_failed);
  another_failed = true;
  refuses([&] { q.end(another_failed); });
  refuses([&] { q.begin(1000, another_failed); });
  require(another_failed);
}
void transitions_refuse_running_calls() {
  std::atomic<bool> failed{false};
  Policy p;
  {
    Guard g("default live", failed, &p);
    refuses<std::logic_error>([&] { p.begin(1000, failed); }, "idle");
  }
  p.begin(1000, failed);
  {
    Guard g("startup live", failed, &p);
    refuses<std::logic_error>([&] { p.end(failed); }, "idle");
  }
  p.end(failed);
}
void two_groups() {
  std::atomic<bool> a_failed{false}, b_failed{false};
  Policy a, b;
  a.begin(120000, a_failed);
  Guard ga("A", a_failed, &a), gb("B", b_failed, &b);
  sample(ga);
  sample(gb);
  FakeClock::ms(30001);
  sample(ga);
  refuses([&] { sample(gb); });
  require(!a_failed && b_failed);
}
void two_active_groups() {
  std::atomic<bool> a_failed{false}, b_failed{false};
  Policy a, b;
  a.begin(1000, a_failed);
  b.begin(2000, b_failed);
  Guard ga("active A", a_failed, &a), gb("active B", b_failed, &b);
  FakeClock::ms(1000);
  refuses([&] { ga.check(&success, 1); });
  gb.check(&success, 1);
  require(!b_failed);
  FakeClock::ms(2000);
  refuses([&] { gb.check(&success, 1); });
}
void shared_wires() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(60000, failed);
  Guard a("wire A", failed, &p), b("wire B", failed, &p);
  FakeClock::ms(60000);
  refuses([&] { a.check(&success, 1); }, "absolute startup");
  refuses<jaccl::WireStopped>([&] { sample(b); });
}
void late_wire_keeps_failure_kind() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(1000, failed);
  failed = true;
  Guard late("late wire", failed, &p);
  refuses<jaccl::WireStopped>([&] { late.check(&success, 1); });
}

void concurrent_begin() {
  std::atomic<bool> failed{false};
  Policy p;
  std::atomic<int> won{0}, refused{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i) {
    threads.emplace_back([&] {
      try {
        p.begin(1000, failed);
        ++won;
      } catch (const std::logic_error&) {
        ++refused;
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  require(won == 1 && refused == 15);
  p.end(failed);
}
void concurrent_transition_and_wires() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(1000, failed);
  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      Guard g("wire", failed, &p);
      ++ready;
      while (!release.load()) {
        std::this_thread::yield();
      }
      g.check(&success, 1);
    });
  }
  while (ready != 8) {
    std::this_thread::yield();
  }
  refuses<std::logic_error>([&] { p.end(failed); }, "idle");
  release = true;
  for (auto& t : threads) {
    t.join();
  }
  p.end(failed);
}
void hardware_error_preserved() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(1000, failed);
  Guard g("hardware", failed, &p);
  ibv_wc wc{};
  wc.status = IBV_WC_WR_FLUSH_ERR;
  refuses([&] { g.check(&wc, 1); }, "work completion failed");
  require(failed);
}
void poll_error_preserved() {
  std::atomic<bool> failed{false};
  Policy p;
  p.begin(1000, failed);
  Guard g("poll", failed, &p);
  refuses([&] { g.check(nullptr, -1); }, "could not be polled");
  require(failed);
}

int environment_policy() {
  std::atomic<bool> failed{false};
  Policy p;
  const int64_t timeout = jaccl::progress_timeout_ms();
  {
    Guard g("environment default", failed, &p);
    sample(g);
    FakeClock::ms(1000000);
    if (timeout > 0) {
      refuses([&] { sample(g); });
    } else {
      sample(g);
      require(!failed);
    }
  }
  failed = false;
  FakeClock::ms(0);
  Policy startup;
  startup.begin(1000, failed);
  {
    Guard g("environment startup", failed, &startup);
    FakeClock::ms(500);
    g.check(&success, 1);
    FakeClock::ms(1000);
    refuses([&] { g.check(&success, 1); }, "absolute startup");
  }
  failed = false;
  FakeClock::ms(0);
  Policy ended;
  ended.begin(1000, failed);
  ended.end(failed);
  {
    Guard g("environment ended", failed, &ended);
    sample(g);
    FakeClock::ms(1000000);
    if (timeout > 0) {
      refuses([&] { sample(g); });
    } else {
      sample(g);
      require(!failed);
    }
  }
  std::cout << "PASS environment default/startup/end timeout=" << timeout
            << "\n";
  return 0;
}

int main(int argc, char**) {
  if (argc > 1) {
    return environment_policy();
  }
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"default_idle", default_idle},
      {"default_progress_resets_idle", default_progress_resets_idle},
      {"startup_deadline", startup_deadline},
      {"progress_cannot_renew", progress_cannot_renew},
      {"deadline_shared_across_calls", deadline_shared_across_calls},
      {"end_restores_idle", end_restores_idle},
      {"invalid_does_not_consume", invalid_does_not_consume},
      {"maximum_budget", maximum_budget},
      {"checked_overflow", checked_overflow},
      {"exact_clock_limit", exact_clock_limit},
      {"negative_clock_epoch", negative_clock_epoch},
      {"one_shot", one_shot},
      {"expired_end_refuses", expired_end_refuses},
      {"closed_group_stays_closed", closed_group_stays_closed},
      {"transitions_refuse_running_calls", transitions_refuse_running_calls},
      {"two_groups", two_groups},
      {"two_active_groups", two_active_groups},
      {"shared_wires", shared_wires},
      {"late_wire_keeps_failure_kind", late_wire_keeps_failure_kind},
      {"concurrent_begin", concurrent_begin},
      {"concurrent_transition_and_wires", concurrent_transition_and_wires},
      {"hardware_error_preserved", hardware_error_preserved},
      {"poll_error_preserved", poll_error_preserved},
  };
  int passed = 0;
  for (const auto& [name, test] : tests) {
    FakeClock::ms(0);
    try {
      test();
      ++passed;
      std::cout << "PASS " << name << "\n";
    } catch (const std::exception& e) {
      std::cerr << "FAIL " << name << ": " << e.what() << "\n";
      return 1;
    }
  }
  std::cout << "passed=" << passed << " failed=0 skipped=0 native_rdma=0\n";
}
