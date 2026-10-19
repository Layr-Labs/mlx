// Copyright © 2026 Apple Inc.
#include <iostream>
#include <type_traits>
#include "jaccl/group.h"
#include "mlx/distributed/distributed_impl.h"

using mlx::core::array;
using mlx::core::Stream;
using mlx::core::StreamOrDevice;
using mlx::core::distributed::Group;
using mlx::core::distributed::detail::GroupImpl;
static_assert(std::is_same_v<
              decltype(&Group::begin_startup_wait_budget),
              void (Group::*)(int64_t) const>);
static_assert(std::is_same_v<
              decltype(&Group::end_startup_wait_budget),
              void (Group::*)() const>);
class Unsupported : public GroupImpl {
 public:
  Stream communication_stream(StreamOrDevice) override {
    throw std::runtime_error("unused");
  }
  int rank() override {
    return 0;
  }
  int size() override {
    return 1;
  }
  std::shared_ptr<GroupImpl> split(int, int) override {
    return nullptr;
  }
  void all_sum(const array&, array&, Stream) override {}
  void all_max(const array&, array&, Stream) override {}
  void all_min(const array&, array&, Stream) override {}
  void all_gather(const array&, array&, Stream) override {}
  void sum_scatter(const array&, array&, Stream) override {}
  void send(const array&, int, Stream) override {}
  void recv(array&, int, Stream) override {}
};
class UnsupportedStandalone : public jaccl::Group {
 public:
  int rank() override {
    return 0;
  }
  int size() override {
    return 1;
  }
  void all_sum(const void*, void*, size_t, int) override {}
  void all_max(const void*, void*, size_t, int) override {}
  void all_min(const void*, void*, size_t, int) override {}
  void all_gather(const void*, void*, size_t) override {}
  void sum_scatter(const void*, void*, size_t, int) override {}
  void send(const void*, size_t, int) override {}
  void recv(void*, size_t, int) override {}
  void barrier() override {}
};
template <typename Fn>
void refuses(Fn fn) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("unsupported backend accepted startup budget");
}
int main() {
  Unsupported group;
  UnsupportedStandalone standalone;
  refuses([&] { group.begin_startup_wait_budget(1000); });
  refuses([&] { group.end_startup_wait_budget(); });
  refuses([&] { standalone.begin_startup_wait_budget(1000); });
  refuses([&] { standalone.end_startup_wait_budget(); });
  std::cout
      << "PASS exact public const API types\nPASS unsupported GroupImpl begin/end\nPASS unsupported standalone begin/end\npassed=3 failed=0 skipped=0 native_rdma=0\n";
}
