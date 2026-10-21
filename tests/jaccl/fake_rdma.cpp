// Copyright © 2026 Apple Inc.

#include <optional>
#include <utility>

#include "fake_rdma.h"

namespace {
std::vector<char> bytes(const ibv_sge& entry) {
  const auto* pointer = reinterpret_cast<const char*>(entry.addr);
  return {pointer, pointer + entry.length};
}

void check_unchanged(const fixture::Pending& pending) {
  fixture::require(
      bytes(pending.entry) == pending.original,
      "Send buffer reused before its completion was polled");
}

void deliver(
    fixture::Endpoint& endpoint,
    int index,
    int drop,
    int fail,
    fixture::Completion completion) {
  if (index == drop) {
    return;
  }
  if (index == fail) {
    completion.value.status = IBV_WC_WR_FLUSH_ERR;
  }
  endpoint.completions.push_back(std::move(completion));
  auto& high_water = fixture::ledger().completion_high_water;
  high_water = std::max(high_water.load(), endpoint.completions.size());
}

void drop(ibv_qp* pair) {
  if (pair) {
    delete pair;
    fixture::ledger().queue_pairs--;
  }
}

void drop(ibv_cq* queue) {
  if (queue) {
    delete queue;
    fixture::ledger().completion_queues--;
  }
}

void drop(ibv_mr* region) {
  if (region) {
    delete region;
    fixture::ledger().regions--;
  }
}

// The three verbs that a failed group calls through jaccl::ibv().
int destroy_queue_pair(ibv_qp* pair) {
  fixture::ledger().destroy_queue_pair_calls++;
  if (fixture::ledger().refuse_destroy) {
    return 16;
  }
  auto& endpoint = *static_cast<fixture::Endpoint*>(pair->fixture);
  {
    // The posted work is cancelled. The peer cannot match this endpoint and
    // its completions are gone.
    std::lock_guard lock(endpoint.fabric->mutex);
    endpoint.closed = true;
    endpoint.sends.clear();
    endpoint.receives.clear();
    endpoint.completions.clear();
  }
  drop(pair);
  return 0;
}

int destroy_completion_queue(ibv_cq* queue) {
  fixture::ledger().destroy_completion_queue_calls++;
  if (fixture::ledger().refuse_destroy) {
    return 16;
  }
  drop(queue);
  return 0;
}

int deregister(ibv_mr* region) {
  fixture::ledger().deregister_calls++;
  drop(region);
  return 0;
}
} // namespace

namespace jaccl {
// Only allocation and registration are fake. The staging methods of
// SharedBuffer and the post and poll methods of Connection are the real ones.
SharedBuffer::SharedBuffer(size_t count) : data_(nullptr), num_bytes_(count) {
  if (posix_memalign(&data_, 4096, count) != 0) {
    throw std::bad_alloc();
  }
  std::memset(data_, fixture::sentinel, count);
}

SharedBuffer::SharedBuffer(SharedBuffer&& other)
    : data_(nullptr), num_bytes_(0) {
  std::swap(data_, other.data_);
  std::swap(num_bytes_, other.num_bytes_);
  memory_regions_.swap(other.memory_regions_);
}

SharedBuffer::~SharedBuffer() {
  for (const auto& entry : memory_regions_) {
    drop(entry.second);
  }
  std::free(data_);
}

void SharedBuffer::register_to_protection_domain(ibv_pd* domain) {
  if (!memory_regions_.contains(domain)) {
    memory_regions_[domain] = new ibv_mr{1};
    fixture::ledger().regions++;
  }
}

Connection::Connection(ibv_context* context)
    : ctx(context),
      protection_domain(nullptr),
      completion_queue(nullptr),
      queue_pair(nullptr),
      src{} {}
Connection::Connection(Connection&& other) : Connection(nullptr) {
  std::swap(ctx, other.ctx);
  std::swap(protection_domain, other.protection_domain);
  std::swap(completion_queue, other.completion_queue);
  std::swap(queue_pair, other.queue_pair);
  std::swap(src, other.src);
  std::swap(source_gid_index, other.source_gid_index);
}

Connection::~Connection() {
  delete ctx;
  delete protection_domain;
  drop(completion_queue);
  drop(queue_pair);
}

// The real wrapper loads librdma.dylib. The fake one has only the three verbs
// that a failed group calls.
IBVWrapper::IBVWrapper() : librdma_handle_(nullptr) {
  get_device_list = nullptr;
  get_device_name = nullptr;
  open_device = nullptr;
  free_device_list = nullptr;
  close_device = nullptr;
  alloc_pd = nullptr;
  create_qp = nullptr;
  create_cq = nullptr;
  dealloc_pd = nullptr;
  query_port = nullptr;
  query_gid = nullptr;
  modify_qp = nullptr;
  reg_mr = nullptr;
  destroy_qp = destroy_queue_pair;
  destroy_cq = destroy_completion_queue;
  dereg_mr = deregister;
}

IBVWrapper& ibv() {
  static IBVWrapper wrapper;
  return wrapper;
}
} // namespace jaccl

namespace fixture {
Ledger& ledger() {
  static Ledger value;
  return value;
}

Endpoint& endpoint(const jaccl::Connection& connection) {
  return *static_cast<Endpoint*>(connection.queue_pair->fixture);
}

void Fabric::pair(
    jaccl::Connection& a,
    jaccl::Connection& b,
    int wire,
    int side) {
  auto first = std::make_unique<Endpoint>();
  auto second = std::make_unique<Endpoint>();
  first->fabric = second->fabric = this;
  first->rank = 0;
  second->rank = 1;
  first->side = side;
  second->side = 1 - side;
  first->wire = second->wire = wire;
  first->peer = second.get();
  second->peer = first.get();
  auto attach = [](jaccl::Connection& connection, Endpoint* endpoint) {
    connection.ctx = new ibv_context{};
    connection.protection_domain = new ibv_pd{};
    connection.completion_queue = new ibv_cq{endpoint};
    connection.queue_pair = new ibv_qp{endpoint};
    ledger().completion_queues++;
    ledger().queue_pairs++;
  };
  attach(a, first.get());
  attach(b, second.get());
  endpoints.push_back(std::move(first));
  endpoints.push_back(std::move(second));
}

void Fabric::match(Endpoint& sender) {
  auto& receiver = *sender.peer;
  while (!sender.closed && !receiver.closed && !sender.sends.empty() &&
         !receiver.receives.empty()) {
    auto send = std::move(sender.sends.front());
    sender.sends.pop_front();
    auto receive = std::move(receiver.receives.front());
    receiver.receives.pop_front();
    check_unchanged(send);
    require(
        send.entry.length == receive.entry.length,
        "The posted send and receive lengths differ");
    std::memcpy(
        reinterpret_cast<void*>(receive.entry.addr),
        reinterpret_cast<const void*>(send.entry.addr),
        send.entry.length);
    frames.push_back({sender.rank, sender.side, sender.wire, send.original});
    const ibv_wc sent{send.id, IBV_WC_SUCCESS, 0, send.entry.length};
    const ibv_wc received{receive.id, IBV_WC_SUCCESS, 0, receive.entry.length};
    deliver(
        sender,
        sender.sent++,
        sender.fault.drop_send,
        sender.fault.fail_send,
        {sent, std::move(send)});
    deliver(
        receiver,
        receiver.received++,
        receiver.fault.drop_receive,
        receiver.fault.fail_receive,
        {received, std::nullopt});
  }
  changed.notify_all();
}

void Fabric::require_drained() {
  std::lock_guard lock(mutex);
  for (const auto& endpoint : endpoints) {
    require(
        endpoint->sends.empty() && endpoint->receives.empty() &&
            endpoint->completions.empty(),
        "A posted request or a completion was not consumed");
  }
}

void Fabric::expire_after(std::chrono::milliseconds duration) {
  std::lock_guard lock(mutex);
  deadline = std::chrono::steady_clock::now() + duration;
}
} // namespace fixture

int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
  fixture::require(
      wr->num_sge == 1 && wr->next == nullptr && wr->opcode == IBV_WR_SEND &&
          wr->send_flags == IBV_SEND_SIGNALED,
      "Unexpected send work request");
  auto& endpoint = *static_cast<fixture::Endpoint*>(qp->fixture);
  std::lock_guard lock(endpoint.fabric->mutex);
  endpoint.sends.push_back({wr->sg_list[0], wr->wr_id, bytes(wr->sg_list[0])});
  endpoint.fabric->match(endpoint);
  return 0;
}

int ibv_post_recv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr**) {
  fixture::require(
      wr->num_sge == 1 && wr->next == nullptr,
      "Unexpected receive work request");
  auto& endpoint = *static_cast<fixture::Endpoint*>(qp->fixture);
  std::lock_guard lock(endpoint.fabric->mutex);
  endpoint.receives.push_back({wr->sg_list[0], wr->wr_id, {}});
  endpoint.fabric->match(*endpoint.peer);
  return 0;
}

int ibv_poll_cq(ibv_cq* cq, int maximum, ibv_wc* out) {
  auto& endpoint = *static_cast<fixture::Endpoint*>(cq->fixture);
  std::unique_lock lock(endpoint.fabric->mutex);
  fixture::require(
      std::chrono::steady_clock::now() < endpoint.fabric->deadline,
      "The fake poll reached its deadline");
  if (endpoint.fault.poll_error) {
    return -1;
  }
  if (endpoint.completions.empty() && endpoint.fabric->idle_wait.count() > 0) {
    endpoint.fabric->changed.wait_for(lock, endpoint.fabric->idle_wait);
  }
  int count = 0;
  while (count < maximum && !endpoint.completions.empty()) {
    auto completion = std::move(endpoint.completions.front());
    endpoint.completions.pop_front();
    if (completion.send) {
      check_unchanged(*completion.send);
    }
    out[count++] = completion.value;
  }
  return count;
}
