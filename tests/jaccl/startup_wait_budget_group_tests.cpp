// Copyright © 2026 Apple Inc.
// CPU simulated verbs only. Real MeshImpl/RingImpl call and cleanup paths.
#include <iostream>
#include <thread>
#include "fixture.h"

void require(bool value) {
  fixture::require(value, "startup group assertion failed");
}
template <typename Fn>
void refuses(Fn fn, const std::string& text) {
  try {
    fn();
  } catch (const std::exception& e) {
    require(std::string(e.what()).find(text) != std::string::npos);
    return;
  }
  throw std::runtime_error("expected refusal");
}
void mesh_success_end() {
  fixture::MeshPair pair;
  for (auto& node : pair.nodes) {
    node->begin_startup_wait_budget(10000);
  }
  char input = 42, output = 0;
  fixture::both(
      [&] { pair.nodes[0]->send(&input, 1, 1); },
      [&] { pair.nodes[1]->recv(&output, 1, 0); });
  require(input == output);
  pair.fabric.require_drained();
  for (auto& node : pair.nodes) {
    node->end_startup_wait_budget();
    refuses([&] { node->begin_startup_wait_budget(10000); }, "only begin once");
  }
  fixture::both(
      [&] { pair.nodes[0]->send(&input, 1, 1); },
      [&] { pair.nodes[1]->recv(&output, 1, 0); });
  pair.fabric.require_drained();
}
void ring_success_end(int wires) {
  fixture::RingPair pair(wires);
  for (auto& node : pair.nodes) {
    node->begin_startup_wait_budget(10000);
  }
  std::vector<char> input(99, 42), output(99, 0);
  fixture::both(
      [&] { pair.nodes[0]->send(input.data(), 99, 1, wires); },
      [&] { pair.nodes[1]->recv(output.data(), 99, 0, wires); });
  require(input == output);
  pair.fabric.require_drained();
  for (auto& node : pair.nodes) {
    node->end_startup_wait_budget();
  }
  fixture::both(
      [&] { pair.nodes[0]->send(input.data(), 99, 1, wires); },
      [&] { pair.nodes[1]->recv(output.data(), 99, 0, wires); });
  pair.fabric.require_drained();
}
// The guard samples the startup deadline once in PROGRESS_SAMPLE_POLLS empty
// polls. With the default 1 ms idle wait per empty poll, that is 1024 sleeps,
// which a hosted runner stretches past the fake poll's 8 s deadline. The
// expiry cases poll without a sleep, as the hardware does.
void spin(fixture::Fabric& fabric) {
  fabric.idle_wait = std::chrono::milliseconds(0);
}
void mesh_expiry_cleanup() {
  fixture::MeshPair pair;
  spin(pair.fabric);
  auto& node = *pair.nodes[0];
  node.begin_startup_wait_budget(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  char output = 0;
  const auto qp = fixture::ledger().destroy_queue_pair_calls.load();
  const auto cq = fixture::ledger().destroy_completion_queue_calls.load();
  const auto mr = fixture::ledger().deregister_calls.load();
  refuses([&] { node.recv(&output, 1, 1); }, "absolute startup");
  require(fixture::ledger().destroy_queue_pair_calls == qp + 1);
  require(fixture::ledger().destroy_completion_queue_calls == cq + 1);
  require(fixture::ledger().deregister_calls > mr);
  refuses([&] { node.end_startup_wait_budget(); }, "closed");
  refuses([&] { node.begin_startup_wait_budget(10000); }, "closed");
  refuses([&] { node.recv(&output, 1, 1); }, "closed");
}
void ring_expiry_cleanup(int wires) {
  fixture::RingPair pair(wires);
  spin(pair.fabric);
  auto& node = *pair.nodes[0];
  node.begin_startup_wait_budget(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  std::vector<char> output(99, 0);
  const auto qp = fixture::ledger().destroy_queue_pair_calls.load();
  const auto cq = fixture::ledger().destroy_completion_queue_calls.load();
  const auto mr = fixture::ledger().deregister_calls.load();
  refuses([&] { node.recv(output.data(), 99, 1, wires); }, "absolute startup");
  require(fixture::ledger().destroy_queue_pair_calls == qp + 2 * wires);
  require(fixture::ledger().destroy_completion_queue_calls == cq + 2 * wires);
  require(fixture::ledger().deregister_calls > mr);
  refuses([&] { node.end_startup_wait_budget(); }, "closed");
  refuses([&] { node.begin_startup_wait_budget(10000); }, "closed");
  refuses([&] { node.recv(output.data(), 99, 1, wires); }, "closed");
}
int main() {
  jaccl::set_memory_release(nullptr);
  require(jaccl::failure_action() == jaccl::FailureAction::Throw);
  mesh_success_end();
  std::cout << "PASS mesh success/end/later-serving\n";
  for (int wires : {1, 2}) {
    ring_success_end(wires);
    std::cout << "PASS ring success/end/later-serving wires=" << wires << "\n";
  }
  mesh_expiry_cleanup();
  std::cout << "PASS mesh expiry/cleanup/closed\n";
  for (int wires : {1, 2}) {
    ring_expiry_cleanup(wires);
    std::cout << "PASS ring expiry/cleanup/closed wires=" << wires << "\n";
  }
  require(
      fixture::ledger().queue_pairs == 0 &&
      fixture::ledger().completion_queues == 0 &&
      fixture::ledger().regions == 0);
  std::cout
      << "PASS fake resource ledger empty\npassed=7 failed=0 skipped=0 native_rdma=0\n";
}
