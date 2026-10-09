// Copyright © 2026 Apple Inc.

// Tests of the progress guard with simulated verbs. Each call goes through
// the real mesh and ring headers. The limit is read once in a process, so each
// mode of this program sets the environment first and is one CTest test.

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "fixture.h"
#include "jaccl/progress_guard.h"

namespace {
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

// The limit of this process in milliseconds.
int64_t limit = 0;
// A call may fail this much after the limit. The fake poll then stops it.
constexpr int64_t slack = 1000;
const char* const watchdog = "The fake poll reached its deadline";
const char* const closed =
    "[jaccl] The group is closed after an earlier failure and cannot be used again.";
const int64_t frame = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));

struct Outcome {
  bool threw = false;
  std::string message;
  int64_t ms = 0;
};

template <typename Call>
Outcome attempt(Call call) {
  Outcome outcome;
  const auto start = Clock::now();
  try {
    call();
  } catch (const std::exception& error) {
    outcome.threw = true;
    outcome.message = error.what();
  }
  outcome.ms =
      std::chrono::duration_cast<milliseconds>(Clock::now() - start).count();
  return outcome;
}

template <typename First, typename Second>
std::array<Outcome, 2> together(First first, Second second) {
  auto a = std::async(std::launch::async, [&] { return attempt(first); });
  auto b = std::async(std::launch::async, [&] { return attempt(second); });
  Outcome x = a.get(), y = b.get();
  return {x, y};
}

// Print each kind of error once, as the caller sees it.
void show(bool& shown, const Outcome& outcome) {
  if (!shown) {
    std::cout << "INFO error: " << outcome.message << '\n';
    shown = true;
  }
}

[[noreturn]] void mismatch(const std::string& what, const Outcome& outcome) {
  std::ostringstream text;
  text << what << " (after " << outcome.ms << " ms: "
       << (outcome.threw ? outcome.message : std::string("returned normally"))
       << ")";
  throw std::runtime_error(text.str());
}

void expect_success(const Outcome& outcome) {
  if (outcome.threw) {
    mismatch("normal traffic failed", outcome);
  }
}

// The polling loop must stop by itself, with its own error, near the limit.
void expect_timeout(const Outcome& outcome, const std::string& op) {
  if (!outcome.threw) {
    mismatch("a stalled call returned without an error", outcome);
  }
  if (outcome.message == watchdog) {
    mismatch(
        "the polling loop did not stop by itself; the watchdog of the fake poll ended it",
        outcome);
  }
  if (outcome.message.rfind("[jaccl] " + op + ": no completion for ", 0) != 0) {
    mismatch("not the no-progress error of " + op, outcome);
  }
  if (outcome.ms < limit || outcome.ms > limit + slack) {
    mismatch("the failure is outside [limit, limit + slack]", outcome);
  }
  static bool shown = false;
  show(shown, outcome);
}

// A failed completion must be an error at once; it must not wait for the limit.
void expect_failed_completion(const Outcome& outcome, const std::string& op) {
  if (!outcome.threw) {
    mismatch("a failed completion was accepted as data", outcome);
  }
  if (outcome.message.rfind(
          "[jaccl] " + op + ": a work completion failed with status 5", 0) !=
      0) {
    mismatch("not the failed-completion error of " + op, outcome);
  }
  if (limit > 0 && outcome.ms >= limit / 2) {
    mismatch("the failure was not immediate", outcome);
  }
  static bool shown = false;
  show(shown, outcome);
}

void expect_closed(const Outcome& outcome) {
  if (!outcome.threw || outcome.message != closed) {
    mismatch("a closed group accepted a call", outcome);
  }
  if (outcome.ms > 100) {
    mismatch("a closed group did not refuse at once", outcome);
  }
  static bool shown = false;
  show(shown, outcome);
}

void arm(fixture::Fabric& fabric, int64_t ms = -1) {
  fabric.idle_wait = milliseconds(0);
  fabric.expire_after(milliseconds(ms < 0 ? limit + slack : ms));
}

// The failed rank destroyed its queue pairs and completion queues through the
// verbs and deregistered every region. `before` is the ledger before the call.
struct Counts {
  int queue_pairs, completion_queues, regions, destroy_pairs, destroy_queues,
      deregister;
  static Counts now() {
    const auto& l = fixture::ledger();
    return {
        l.queue_pairs,
        l.completion_queues,
        l.regions,
        l.destroy_queue_pair_calls,
        l.destroy_completion_queue_calls,
        l.deregister_calls};
  }
};

void expect_released(
    const Counts& before,
    std::initializer_list<const std::vector<jaccl::Connection>*> links,
    int ranks_failed,
    int ranks) {
  int connections = 0;
  for (const auto* group : links) {
    for (const auto& connection : *group) {
      if (connection.ctx == nullptr) {
        continue;
      }
      connections++;
      fixture::require(
          connection.queue_pair == nullptr &&
              connection.completion_queue == nullptr,
          "A failed group kept a queue pair or a completion queue");
    }
  }
  const Counts after = Counts::now();
  fixture::require(
      after.destroy_pairs - before.destroy_pairs == connections &&
          after.destroy_queues - before.destroy_queues == connections &&
          before.queue_pairs - after.queue_pairs == connections &&
          before.completion_queues - after.completion_queues == connections,
      "Queue pairs/completion queues were not destroyed exactly once");
  const int regions = before.regions * ranks_failed / ranks;
  fixture::require(
      regions > 0 && before.regions - after.regions == regions &&
          after.deregister - before.deregister == regions,
      "The buffers of a failed group were not deregistered exactly once");
}

std::vector<float> floats(size_t count, float value) {
  return std::vector<float>(count, value);
}

const char* raw(const std::vector<float>& values) {
  return reinterpret_cast<const char*>(values.data());
}

char* raw(std::vector<float>& values) {
  return reinterpret_cast<char*>(values.data());
}

using MeshCall = std::function<void(jaccl::MeshImpl&)>;
using RingCall = std::function<void(jaccl::RingImpl&, int wires)>;
struct Buffers {
  std::vector<float> in = floats(4096, 1), out = floats(8192, -19);
};

std::vector<std::pair<std::string, MeshCall>> mesh_calls(Buffers& b) {
  return {
      {"mesh recv", [&](auto& node) { node.recv(raw(b.out), 100, 1); }},
      {"mesh send", [&](auto& node) { node.send(raw(b.in), 100, 1); }},
      {"mesh all_reduce",
       [&](auto& node) {
         node.all_reduce(b.in.data(), b.out.data(), 64, fixture::Add{});
       }},
      {"mesh all_gather",
       [&](auto& node) { node.all_gather(raw(b.in), raw(b.out), 256); }},
      {"mesh sum_scatter",
       [&](auto& node) {
         node.sum_scatter(b.in.data(), b.out.data(), 64, fixture::Add{});
       }},
      // The scatter-gather all reduce starts with a reduce scatter.
      {"mesh sum_scatter",
       [&](auto& node) {
         node.all_reduce_scatter_gather(
             b.in.data(), b.out.data(), 64, fixture::Add{});
       }},
  };
}

std::vector<std::pair<std::string, RingCall>> ring_calls(Buffers& b) {
  return {
      {"ring recv",
       [&](auto& node, int wires) { node.recv(raw(b.out), 100, 1, wires); }},
      {"ring send",
       [&](auto& node, int wires) { node.send(raw(b.in), 100, 1, wires); }},
      {"ring all_reduce/all_gather",
       [&](auto& node, int wires) {
         if (wires == 1) {
           node.template all_reduce<1>(
               b.in.data(), b.out.data(), 64, 1, fixture::Add{});
         } else {
           node.template all_reduce<2>(
               b.in.data(), b.out.data(), 64, wires, fixture::Add{});
         }
       }},
      {"ring all_reduce/all_gather",
       [&](auto& node, int wires) {
         node.all_gather(raw(b.in), raw(b.out), 256, wires);
       }},
      {"ring reduce_scatter",
       [&](auto& node, int wires) {
         node.reduce_scatter(
             b.in.data(), b.out.data(), 128, wires, fixture::Add{});
       }},
  };
}

int failures = 0, passes = 0;
template <typename Check>
void check(const std::string& name, Check body) {
  try {
    body();
    std::cout << "PASS " << name << '\n';
    passes++;
  } catch (const std::exception& error) {
    std::cout << "FAIL " << name << ": " << error.what() << '\n';
    failures++;
  }
}

// A peer that never answers: every polling loop must stop by itself, release
// the group and refuse every later call.
void silent_peer() {
  Buffers b;
  const auto mesh = mesh_calls(b);
  const char* mesh_names[] = {
      "recv",
      "send",
      "all_reduce",
      "all_gather",
      "sum_scatter",
      "all_reduce_scatter_gather"};
  for (size_t index = 0; index < mesh.size(); ++index) {
    check(
        std::string("silent peer: mesh ") + mesh_names[index] +
            " stops, releases, then refuses",
        [&] {
          fixture::MeshPair pair;
          arm(pair.fabric);
          const Counts before = Counts::now();
          expect_timeout(
              attempt([&] { mesh[index].second(*pair.nodes[0]); }),
              mesh[index].first);
          expect_released(before, {&pair.links[0]}, 1, 2);
          for (const auto& later : mesh) {
            expect_closed(attempt([&] { later.second(*pair.nodes[0]); }));
          }
        });
  }
  const auto ring = ring_calls(b);
  const char* ring_names[] = {
      "recv", "send", "all_reduce", "all_gather", "reduce_scatter"};
  for (int wires : {1, 2}) {
    for (size_t index = 0; index < ring.size(); ++index) {
      check(
          "silent peer: ring " + std::string(ring_names[index]) + " on " +
              std::to_string(wires) + " wire(s) stops, releases, then refuses",
          [&] {
            fixture::RingPair pair(wires);
            arm(pair.fabric);
            const Counts before = Counts::now();
            expect_timeout(
                attempt([&] { ring[index].second(*pair.nodes[0], wires); }),
                ring[index].first);
            expect_released(before, {&pair.left[0], &pair.right[0]}, 1, 2);
            for (const auto& later : ring) {
              expect_closed(
                  attempt([&] { later.second(*pair.nodes[0], wires); }));
            }
          });
    }
  }
}

// One completion is lost while the peer keeps working.
void lost_completion() {
  check(
      "lost completion: mesh all_reduce drain stops after the reduction finished",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric);
        fixture::endpoint(pair.links[0][1]).fault.drop_send = 0;
        std::array<std::vector<float>, 2> in{floats(16, 1), floats(16, 2)},
            out{floats(17, -19), floats(17, -19)};
        const Counts before = Counts::now();
        auto result = together(
            [&] {
              pair.nodes[0]->all_reduce(
                  in[0].data(), out[0].data(), 16, fixture::Add{});
            },
            [&] {
              pair.nodes[1]->all_reduce(
                  in[1].data(), out[1].data(), 16, fixture::Add{});
            });
        expect_timeout(result[0], "mesh all_reduce");
        expect_success(result[1]);
        for (const auto& values : out) {
          fixture::require(
              std::all_of(
                  values.begin(),
                  values.begin() + 16,
                  [](float v) { return v == 3; }) &&
                  values[16] == -19,
              "The reduction before the drain changed");
        }
        expect_released(before, {&pair.links[0]}, 1, 2);
      });
  check(
      "lost completion: mesh recv stops in the middle of a five-frame transfer",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric);
        fixture::endpoint(pair.links[1][0]).fault.drop_receive = 2;
        std::vector<char> in(frame * 5, 0x53), out(frame * 5, 0x29);
        const Counts before = Counts::now();
        auto result = together(
            [&] { pair.nodes[0]->send(in.data(), in.size(), 1); },
            [&] { pair.nodes[1]->recv(out.data(), out.size(), 0); });
        expect_success(result[0]);
        expect_timeout(result[1], "mesh recv");
        expect_released(before, {&pair.links[1]}, 1, 2);
      });
  // Wire 1 runs inline and wire 0 on the pool, so both orders are covered.
  for (int stalled : {0, 1}) {
    check(
        "lost completion: ring send on 2 wires stops when wire " +
            std::to_string(stalled) + " loses its completion",
        [&] {
          fixture::RingPair pair(2);
          arm(pair.fabric);
          // Rank 0 sends to rank 1 on its left connections (direction 1).
          fixture::endpoint(pair.left[0][stalled]).fault.drop_send = 0;
          std::vector<char> in(200, 0x64), out(201, 0x29);
          const Counts before = Counts::now();
          auto result = together(
              [&] { pair.nodes[0]->send(in.data(), 200, 1, 2); },
              [&] { pair.nodes[1]->recv(out.data(), 200, 0, 2); });
          expect_timeout(result[0], "ring send");
          expect_success(result[1]);
          fixture::require(
              std::equal(in.begin(), in.end(), out.begin()) &&
                  out.back() == 0x29,
              "The receiver of the completed transfer changed");
          expect_released(before, {&pair.left[0], &pair.right[0]}, 1, 2);
        });
  }
}

// A completion with a failure status is an error, never data.
void failed_completion() {
  check(
      "failed completion: mesh recv throws at once and leaves the output alone",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric);
        fixture::endpoint(pair.links[1][0]).fault.fail_receive = 0;
        std::vector<char> in(100, 0x53), out(100, 0x29);
        const Counts before = Counts::now();
        auto result = together(
            [&] { pair.nodes[0]->send(in.data(), 100, 1); },
            [&] { pair.nodes[1]->recv(out.data(), 100, 0); });
        expect_success(result[0]);
        expect_failed_completion(result[1], "mesh recv");
        fixture::require(
            std::all_of(
                out.begin(), out.end(), [](char v) { return v == 0x29; }),
            "A failed completion was copied to the output");
        expect_released(before, {&pair.links[1]}, 1, 2);
        Buffers b;
        for (const auto& later : mesh_calls(b)) {
          expect_closed(attempt([&] { later.second(*pair.nodes[1]); }));
        }
      });
  check("failed completion: mesh send throws at once", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric);
    fixture::endpoint(pair.links[0][1]).fault.fail_send = 0;
    std::vector<char> in(100, 0x53), out(100, 0x29);
    auto result = together(
        [&] { pair.nodes[0]->send(in.data(), 100, 1); },
        [&] { pair.nodes[1]->recv(out.data(), 100, 0); });
    expect_failed_completion(result[0], "mesh send");
    expect_success(result[1]);
  });
  Buffers b;
  const auto mesh = mesh_calls(b);
  for (size_t index : {size_t(2), size_t(3), size_t(4)}) {
    check("failed completion: " + mesh[index].first + " throws at once", [&] {
      fixture::MeshPair pair;
      arm(pair.fabric);
      fixture::endpoint(pair.links[0][1]).fault.fail_receive = 0;
      Buffers peer;
      const auto peer_calls = mesh_calls(peer);
      auto result = together(
          [&] { mesh[index].second(*pair.nodes[0]); },
          [&] { peer_calls[index].second(*pair.nodes[1]); });
      expect_failed_completion(result[0], mesh[index].first);
      expect_success(result[1]);
    });
  }
  check("failed poll: mesh recv throws at once", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric);
    fixture::endpoint(pair.links[1][0]).fault.poll_error = true;
    std::vector<char> out(100, 0x29);
    const Outcome outcome =
        attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0); });
    if (!outcome.threw ||
        outcome.message.rfind(
            "[jaccl] mesh recv: the completion queue could not be polled", 0) !=
            0 ||
        outcome.ms > 100) {
      mismatch("a failed poll was not an immediate error", outcome);
    }
    static bool shown = false;
    show(shown, outcome);
  });
  // The collectives poll several connections through the shared helpers.
  check("failed poll: mesh all_reduce throws at once", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric);
    fixture::endpoint(pair.links[0][1]).fault.poll_error = true;
    Buffers b;
    const Counts before = Counts::now();
    const Outcome outcome = attempt([&] {
      pair.nodes[0]->all_reduce(b.in.data(), b.out.data(), 64, fixture::Add{});
    });
    if (!outcome.threw ||
        outcome.message.rfind(
            "[jaccl] mesh all_reduce: the completion queue could not be polled",
            0) != 0 ||
        outcome.ms > 100) {
      mismatch("a failed poll was not an immediate error", outcome);
    }
    expect_released(before, {&pair.links[0]}, 1, 2);
  });
  for (int side : {0, 1}) {
    check(
        std::string("failed poll: ring all_reduce throws at once (") +
            (side ? "right" : "left") + " connection)",
        [&] {
          fixture::RingPair pair(1);
          arm(pair.fabric);
          fixture::endpoint(side ? pair.right[0][0] : pair.left[0][0])
              .fault.poll_error = true;
          Buffers b;
          const Counts before = Counts::now();
          const Outcome outcome = attempt([&] {
            pair.nodes[0]->all_reduce<1>(
                b.in.data(), b.out.data(), 64, 1, fixture::Add{});
          });
          if (!outcome.threw ||
              outcome.message.rfind(
                  "[jaccl] ring all_reduce/all_gather: the completion queue could not be polled",
                  0) != 0 ||
              outcome.ms > 100) {
            mismatch("a failed poll was not an immediate error", outcome);
          }
          expect_released(before, {&pair.left[0], &pair.right[0]}, 1, 2);
        });
  }
  // Two wires: one gets the failed completion, the other has no traffic. The
  // idle wire must stop, the cause must be reported, and it must not take the
  // limit. `failing` selects the wire; wire 1 runs inline, wire 0 on the pool.
  for (int failing : {0, 1}) {
    check(
        "failed completion: ring recv on 2 wires, wire " +
            std::to_string(failing) +
            " fails, the idle wire stops and the cause is reported",
        [&] {
          fixture::RingPair pair(2);
          arm(pair.fabric);
          // Rank 1 receives from rank 0 on its right connections (direction 1).
          fixture::endpoint(pair.right[1][failing]).fault.fail_receive = 0;
          std::vector<char> in(100, 0x64), out(100, 0x29);
          const Counts before = Counts::now();
          auto result = together(
              [&] { pair.nodes[0]->send_wire(in.data(), 100, 1, 50, failing); },
              [&] { pair.nodes[1]->recv(out.data(), 100, 0, 2); });
          expect_success(result[0]);
          expect_failed_completion(result[1], "ring recv");
          fixture::require(
              std::all_of(
                  out.begin(), out.end(), [](char v) { return v == 0x29; }),
              "A failed completion was copied to the output");
          expect_released(before, {&pair.left[1], &pair.right[1]}, 1, 2);
          Buffers b;
          for (const auto& later : ring_calls(b)) {
            expect_closed(attempt([&] { later.second(*pair.nodes[1], 2); }));
          }
        });
  }
  check(
      "failed completion: ring all_reduce on 2 wires fails on one rank, the peer then stops by itself",
      [&] {
        fixture::RingPair pair(2);
        arm(pair.fabric);
        fixture::endpoint(pair.left[0][0]).fault.fail_receive = 0;
        std::array<std::vector<float>, 2> in{floats(64, 1), floats(64, 2)},
            out{floats(96, -19), floats(96, -19)};
        const Counts before = Counts::now();
        auto result = together(
            [&] {
              pair.nodes[0]->all_reduce<2>(
                  in[0].data(), out[0].data(), 64, 2, fixture::Add{});
            },
            [&] {
              pair.nodes[1]->all_reduce<2>(
                  in[1].data(), out[1].data(), 64, 2, fixture::Add{});
            });
        expect_failed_completion(result[0], "ring all_reduce/all_gather");
        expect_timeout(result[1], "ring all_reduce/all_gather");
        expect_released(
            before,
            {&pair.left[0], &pair.right[0], &pair.left[1], &pair.right[1]},
            2,
            2);
      });
}

// When the verbs refuse to destroy, the failure is still reported, the call
// still fails, and the group still refuses later calls.
void refused_destroy() {
  check(
      "refused destroy: the error is printed, the handles stay for the destructor",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric);
        fixture::endpoint(pair.links[1][0]).fault.poll_error = true;
        std::vector<char> out(100, 0x29);
        std::ostringstream captured;
        auto* previous = std::cerr.rdbuf(captured.rdbuf());
        fixture::ledger().refuse_destroy = true;
        const Outcome outcome =
            attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0); });
        const Outcome later =
            attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0); });
        fixture::ledger().refuse_destroy = false;
        std::cerr.rdbuf(previous);
        if (!outcome.threw ||
            outcome.message.rfind("[jaccl] mesh recv: ", 0) != 0) {
          mismatch("the call did not fail", outcome);
        }
        expect_closed(later);
        fixture::require(
            captured.str() ==
                "[jaccl] Could not destroy a queue pair (16)\n"
                "[jaccl] Could not destroy a completion queue (16)\n",
            "The refused destroy was not reported");
        fixture::require(
            pair.links[1][0].queue_pair != nullptr &&
                pair.links[1][0].completion_queue != nullptr,
            "A handle that was not destroyed was dropped");
      });
}

// Traffic that waits, but less than the limit, is not touched.
void below_limit() {
  check("below the limit: a peer that starts late is waited for", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric, 8000);
    std::vector<char> in(100, 0x53), out(101, 0x29);
    auto result = together(
        [&] {
          std::this_thread::sleep_for(milliseconds(limit / 2));
          pair.nodes[0]->send(in.data(), 100, 1);
        },
        [&] { pair.nodes[1]->recv(out.data(), 100, 0); });
    expect_success(result[0]);
    expect_success(result[1]);
    fixture::require(
        std::equal(in.begin(), in.end(), out.begin()) && out.back() == 0x29,
        "Late transfer changed");
    pair.fabric.require_drained();
  });
  check(
      "below the limit: three gaps longer than the limit in total, each one shorter",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric, 8000);
        std::vector<char> in(frame * 3), out(frame * 3 + 1, 0x29);
        for (size_t index = 0; index < in.size(); ++index) {
          in[index] = char(index / frame + 1);
        }
        auto result = together(
            [&] {
              for (int part = 0; part < 3; ++part) {
                std::this_thread::sleep_for(milliseconds(limit / 2));
                pair.nodes[0]->send(in.data() + part * frame, frame, 1);
              }
            },
            [&] { pair.nodes[1]->recv(out.data(), frame * 3, 0); });
        expect_success(result[0]);
        expect_success(result[1]);
        fixture::require(
            result[1].ms >= limit * 3 / 2,
            "The receiver did not wait across the gaps");
        fixture::require(
            std::equal(in.begin(), in.end(), out.begin()) && out.back() == 0x29,
            "Paused transfer changed");
        pair.fabric.require_drained();
      });
  check(
      "below the limit: ring all_reduce on 2 wires with a peer that starts late",
      [&] {
        fixture::RingPair pair(2);
        arm(pair.fabric, 8000);
        std::array<std::vector<float>, 2> in{floats(4096, 1), floats(4096, 2)},
            out{floats(4097, -19), floats(4097, -19)};
        auto result = together(
            [&] {
              pair.nodes[0]->all_reduce<2>(
                  in[0].data(), out[0].data(), 4096, 2, fixture::Add{});
            },
            [&] {
              std::this_thread::sleep_for(milliseconds(limit / 2));
              pair.nodes[1]->all_reduce<2>(
                  in[1].data(), out[1].data(), 4096, 2, fixture::Add{});
            });
        expect_success(result[0]);
        expect_success(result[1]);
        for (const auto& values : out) {
          fixture::require(
              std::all_of(
                  values.begin(),
                  values.begin() + 4096,
                  [](float v) { return v == 3; }) &&
                  values[4096] == -19,
              "Late ring reduction changed");
        }
        pair.fabric.require_drained();
      });
}

// With the limit removed the loops wait as before; a failed completion is
// still an error.
void disabled() {
  check(
      "disabled: a stalled mesh recv keeps waiting until the watchdog of the fake poll",
      [&] {
        fixture::MeshPair pair;
        arm(pair.fabric, 1200);
        std::vector<char> out(100, 0x29);
        const Outcome outcome =
            attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0); });
        if (!outcome.threw || outcome.message != watchdog ||
            outcome.ms < 1200) {
          mismatch("the loop stopped although the limit is removed", outcome);
        }
      });
  check(
      "disabled: a stalled ring recv on 2 wires keeps waiting until the watchdog of the fake poll",
      [&] {
        fixture::RingPair pair(2);
        arm(pair.fabric, 1200);
        std::vector<char> out(100, 0x29);
        const Outcome outcome =
            attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0, 2); });
        if (!outcome.threw || outcome.message != watchdog ||
            outcome.ms < 1200) {
          mismatch("the loop stopped although the limit is removed", outcome);
        }
      });
  check("disabled: a failed completion is still an immediate error", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric, 8000);
    fixture::endpoint(pair.links[1][0]).fault.fail_receive = 0;
    std::vector<char> in(100, 0x53), out(100, 0x29);
    auto result = together(
        [&] { pair.nodes[0]->send(in.data(), 100, 1); },
        [&] { pair.nodes[1]->recv(out.data(), 100, 0); });
    expect_success(result[0]);
    expect_failed_completion(result[1], "mesh recv");
    if (result[1].ms > 500) {
      mismatch("the failure was not immediate", result[1]);
    }
  });
}

// Check the limit that the guard read from the environment of this process.
void value(int64_t expected) {
  limit = expected;
  check(
      "the limit of this process is " + std::to_string(expected) + " ms", [&] {
        if (jaccl::progress_timeout_ms() != expected) {
          throw std::runtime_error(
              "the guard read " + std::to_string(jaccl::progress_timeout_ms()));
        }
      });
}

// One stalled call that must stop at the limit of this process.
void stall() {
  check("silent peer: mesh recv stops at the limit of this process", [&] {
    fixture::MeshPair pair;
    arm(pair.fabric);
    std::vector<char> out(100, 0x29);
    expect_timeout(
        attempt([&] { pair.nodes[1]->recv(out.data(), 100, 0); }), "mesh recv");
  });
}

} // namespace

// One mode in each process:
//   checks        all cases with a limit of 250 ms
//   disabled      a limit of 0: a stalled call continues to wait
//   alias         the limit comes from MLX_JACCL_PROGRESS_TIMEOUT_MS
//   precedence    JACCL_PROGRESS_TIMEOUT_MS is used before the alias
//   negative      a negative value removes the limit
//   not-a-number  a value that is not a number keeps the default
//   default       no variable: the default
int main(int count, char** arguments) {
  const char* name = "JACCL_PROGRESS_TIMEOUT_MS";
  const char* alias = "MLX_JACCL_PROGRESS_TIMEOUT_MS";
  const std::string mode = count == 2 ? arguments[1] : "";
  // The guard reads the variables when the first call starts.
  unsetenv(name);
  unsetenv(alias);
  if (mode == "checks") {
    setenv(name, "250", 1);
    value(250);
    silent_peer();
    lost_completion();
    failed_completion();
    refused_destroy();
    below_limit();
  } else if (mode == "disabled") {
    setenv(name, "0", 1);
    value(0);
    disabled();
  } else if (mode == "alias") {
    setenv(alias, "400", 1);
    value(400);
    stall();
  } else if (mode == "precedence") {
    setenv(name, "300", 1);
    setenv(alias, "60000", 1);
    value(300);
    stall();
  } else if (mode == "negative") {
    setenv(name, "-1", 1);
    value(-1);
  } else if (mode == "not-a-number") {
    setenv(name, "30s", 1);
    value(jaccl::DEFAULT_PROGRESS_TIMEOUT_MS);
  } else if (mode == "default") {
    value(jaccl::DEFAULT_PROGRESS_TIMEOUT_MS);
  } else {
    std::cout << "FAIL usage: progress_guard_tests checks | disabled | alias | "
                 "precedence | negative | not-a-number | default\n";
    return 2;
  }
  const auto& ledger = fixture::ledger();
  check(
      "every fake queue pair, completion queue and region was freed once", [&] {
        fixture::require(
            ledger.queue_pairs == 0 && ledger.completion_queues == 0 &&
                ledger.regions == 0,
            "Fake verbs objects remain after the checks");
      });
  std::cout << (failures ? "FAIL " : "PASS ") << passes << " passed, "
            << failures << " failed; limit " << limit
            << " ms; simulated verbs\n";
  return failures ? 1 : 0;
}
