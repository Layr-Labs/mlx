// Copyright © 2026 Apple Inc.

// Simulated verbs for the tests. A send completes when it meets a posted
// receive of the peer endpoint. Each endpoint can lose or fail one completion.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>

#include "jaccl/rdma.h"

namespace fixture {

constexpr unsigned char sentinel = 0xa7;
struct Frame {
  int rank, side, wire;
  std::vector<char> bytes;
};

struct Pending {
  ibv_sge entry;
  uint64_t id;
  std::vector<char> original;
};

struct Completion {
  ibv_wc value;
  std::optional<Pending> send;
};

// The index, from 0, of the completion that an endpoint loses or fails.
struct Fault {
  int drop_send = -1, fail_send = -1, drop_receive = -1, fail_receive = -1;
  bool poll_error = false;
};

// The fake verbs objects that exist and the calls made through jaccl::ibv().
struct Ledger {
  std::atomic<int> queue_pairs{0}, completion_queues{0}, regions{0};
  std::atomic<int> destroy_queue_pair_calls{0},
      destroy_completion_queue_calls{0};
  std::atomic<int> deregister_calls{0};
  std::atomic<size_t> completion_high_water{0};
  std::atomic<bool> refuse_destroy{false};
};

Ledger& ledger();
struct Endpoint;
struct Fabric {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::unique_ptr<Endpoint>> endpoints;
  std::vector<Frame> frames;
  // The fake poll throws after this time, so that no test can run forever.
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(8);
  // The sleep of an empty poll. With 0 the polling loop spins, as it does on
  // hardware.
  std::chrono::milliseconds idle_wait{1};
  void
  pair(jaccl::Connection&, jaccl::Connection&, int wire, int first_side = 0);
  void match(Endpoint&);
  void require_drained();
  void expire_after(std::chrono::milliseconds);
};

struct Endpoint {
  Fabric* fabric;
  Endpoint* peer = nullptr;
  int rank, side, wire;
  std::deque<Pending> sends, receives;
  std::deque<Completion> completions;
  Fault fault;
  int sent = 0, received = 0;
  bool closed = false;
};

// Valid only while the connection has its queue pair.
Endpoint& endpoint(const jaccl::Connection&);
inline void require(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}
} // namespace fixture
