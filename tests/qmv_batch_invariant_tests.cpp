// Copyright © 2026 Eigen Labs.
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/utils.h"

using namespace mlx::core;

namespace {

// A copy of get_qmv_batch_limit (backend/metal/quantized.cpp): the first row
// count of x @ w.T (K inputs, N outputs) that takes a matrix kernel.
int qmv_batch_limit(int K, int N) {
  auto& d = metal::device(Device::gpu);
  char size = d.get_architecture().back();
  int gen = d.get_architecture_gen();
  bool small = K <= 2048 && N <= 2048;
  bool medium = K <= 4096 && N <= 4096;
  if (size != 'd' && gen >= 17) {
    return small ? 33 : (medium ? 25 : 13);
  }
  if (size != 'd' && gen >= 15) {
    return small ? 13 : (medium ? 15 : 13);
  }
  if (size == 'd') {
    return small ? 32 : (medium ? 18 : 12);
  }
  if (gen >= 13) {
    return small ? 14 : (medium ? 10 : ((gen == 13 && size == 's') ? 7 : 6));
  }
  return small ? 18 : (medium ? 12 : 10);
}

struct Weights {
  std::string mode;
  int group_size;
  array w;
  array scales;
  std::optional<array> biases;
};

// Synthetic 4-bit weights with N outputs and K inputs: random packed values,
// affine in groups of 64 with bfloat16 scales and biases, or nvfp4 with
// positive e4m3 scales (bytes 0x20 to 0x4f).
Weights weights(const std::string& mode, int N, int K, uint64_t seed) {
  auto key = [&](int i) { return random::key(4 * seed + i); };
  auto w = random::bits({N, K / 8}, 4, key(0), Device::gpu);
  if (mode == "affine") {
    auto values = [&](int i, float scale) {
      return astype(
          random::normal(
              {N, K / 64}, float32, 0.0f, scale, key(i), Device::gpu),
          bfloat16,
          Device::gpu);
    };
    return {mode, 64, w, values(1, 0.02f), values(2, 0.05f)};
  }
  auto scales = random::randint(
      array(0x20), array(0x50), {N, K / 16}, uint8, key(1), Device::gpu);
  return {mode, 16, w, scales, std::nullopt};
}

array product(const Weights& q, const array& x) {
  return quantized_matmul(
      x, q.w, q.scales, q.biases, true, q.group_size, 4, q.mode, Device::gpu);
}

// Each row of x [R, K] as its own product in one launch: x [R, 1, K] and the
// weights broadcast to [R, N, K / 8], so M = 1 and the batch is R (the
// decode rows of a batch of requests).
array single_rows(const Weights& q, const array& x) {
  auto lead = [](const array& a) { return expand_dims(a, 0, Device::gpu); };
  auto y = quantized_matmul(
      expand_dims(x, 1, Device::gpu),
      lead(q.w),
      lead(q.scales),
      q.biases ? std::optional<array>(lead(*q.biases)) : std::nullopt,
      true,
      q.group_size,
      4,
      q.mode,
      Device::gpu);
  return squeeze(y, 1, Device::gpu);
}

array rows(const array& a, int start, int stop) {
  return slice(a, {start, 0}, {stop, a.shape(1)}, Device::gpu);
}

// Compares the bytes, so that 0 and -0 differ and two NaNs can be equal.
bool same_bytes(const array& a, const array& b) {
  auto bytes = [](const array& x) {
    auto type = x.dtype() == float32 ? uint32 : uint16;
    return view(contiguous(x, false, Device::gpu), type, Device::gpu);
  };
  return array_equal(bytes(a), bytes(b), Device::cpu).item<bool>();
}

} // namespace

TEST_CASE("quantized matmul rows keep their bits below the vector limit") {
  // MLX reads MLX_QMV_BATCH_INVARIANT once in each process. The tests binary
  // runs this case without it, and the qmv_batch_invariant ctest entry runs
  // it again with MLX_QMV_BATCH_INVARIANT=1 (tests/CMakeLists.txt).
  const char* value = std::getenv("MLX_QMV_BATCH_INVARIANT");
  bool invariant = value != nullptr && std::atoi(value) != 0;
  REQUIRE(env::qmv_batch_invariant() == invariant);

  // Tensor-parallel shard shapes (N outputs, K inputs) of GLM and Qwen
  // projections.
  std::vector<std::pair<int, int>> shapes = {
      {4096, 4096},
      {6144, 4096},
      {4096, 6144},
      {8704, 5120},
      {5120, 8704},
      {3072, 5120},
      {2048, 4096},
      {1024, 4096}};
  for (auto [N, K] : shapes) {
    // With the switch, every row count below the vector limit. Without it,
    // up to 5 rows: from 6 rows other kernels can serve some shapes
    // (qmv_mma from 6 rows, the NAX split-K from 8 rows).
    int limit = qmv_batch_limit(K, N);
    int count = invariant ? limit - 1 : std::min(5, limit - 1);
    for (auto mode : {"affine", "nvfp4"}) {
      auto q = weights(mode, N, K, N + K);
      for (auto dtype : {bfloat16, float32}) {
        CAPTURE(N);
        CAPTURE(K);
        CAPTURE(mode);
        CAPTURE(dtype);
        CAPTURE(count);
        auto x = astype(
            random::normal(
                {count, K}, float32, 0.0f, 1.0f, random::key(K), Device::gpu),
            dtype,
            Device::gpu);
        auto all = product(q, x);
        eval(all);
        // A window of m rows alone (one request) has the bits of the same
        // rows among all the rows (several requests in one launch), at the
        // start and at the end. From 2 rows the vector kernel is qmv_wide
        // (or qmv where qmv_wide does not serve the mode on this GPU).
        for (int m = 2; m <= count; ++m) {
          CAPTURE(m);
          CHECK(same_bytes(product(q, rows(x, 0, m)), rows(all, 0, m)));
          CHECK(same_bytes(
              product(q, rows(x, count - m, count)),
              rows(all, count - m, count)));
        }
        // One row is qmv: alone, it has the bits that it has in a batch of
        // single rows. It can have other bits among more rows.
        auto singles = single_rows(q, x);
        for (int i = 0; i < count; ++i) {
          CAPTURE(i);
          CHECK(same_bytes(
              product(q, rows(x, i, i + 1)), rows(singles, i, i + 1)));
        }
      }
    }
  }
}
