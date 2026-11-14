// Copyright © 2026 Eigen Labs.
#include <cstdlib>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/utils.h"

using namespace mlx::core;

namespace {

struct Weights {
  int group_size;
  int bits;
  array w;
  array scales;
  array biases;
};

// Synthetic affine weights with N outputs and K inputs: random packed values,
// and scales and biases with the type of x.
Weights
weights(int N, int K, int group_size, int bits, Dtype dtype, uint64_t seed) {
  auto key = [&](int i) { return random::key(4 * seed + i); };
  auto w = random::bits({N, K * bits / 32}, 4, key(0), Device::gpu);
  auto values = [&](int i, float scale) {
    return astype(
        random::normal(
            {N, K / group_size}, float32, 0.0f, scale, key(i), Device::gpu),
        dtype,
        Device::gpu);
  };
  return {group_size, bits, w, values(1, 0.02f), values(2, 0.05f)};
}

array product(const Weights& q, const array& x) {
  return quantized_matmul(
      x,
      q.w,
      q.scales,
      q.biases,
      true,
      q.group_size,
      q.bits,
      "affine",
      Device::gpu);
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

TEST_CASE("quantized matmul rows keep the bits of one row") {
  // MLX reads MLX_QMV_ROW_EXACT once in each process. The tests binary runs
  // this case without it, and then there is no row-exact product to compare.
  // The qmv_row_exact ctest entry runs it with MLX_QMV_ROW_EXACT=16
  // (tests/CMakeLists.txt).
  const char* value = std::getenv("MLX_QMV_ROW_EXACT");
  int limit = value != nullptr ? std::atoi(value) : 0;
  REQUIRE(env::qmv_row_exact() == limit);
  if (limit < 2) {
    return;
  }

  // N outputs and K inputs. K selects the one-row kernel: the fast kernel (K
  // a multiple of 512 for 4 bits and of 256 for 8 bits), the general kernel
  // (also with a short last block), or qmv_quad (K = 64 or 128). Some N are
  // not a multiple of the 8 output rows of a threadgroup.
  std::vector<std::pair<int, int>> shapes = {
      {2048, 4096},
      {64, 1024},
      {61, 5120},
      {2, 1536},
      {1, 2048},
      {1000, 768},
      {61, 384},
      {5, 320},
      {3, 96},
      {7, 32},
      {9, 128},
      {64, 64}};
  for (auto [N, K] : shapes) {
    for (int bits : {2, 4, 8}) {
      for (int group_size : {32, 64, 128}) {
        // 2 bits are served where the one-row kernel is the fast one.
        if (K % group_size != 0 || (bits == 2 && K % 512 != 0)) {
          continue;
        }
        for (auto dtype : {float32, float16, bfloat16}) {
          CAPTURE(N);
          CAPTURE(K);
          CAPTURE(bits);
          CAPTURE(group_size);
          CAPTURE(dtype);
          auto q = weights(N, K, group_size, bits, dtype, N + K + bits);
          auto x = astype(
              random::normal(
                  {limit, K}, float32, 0.0f, 1.0f, random::key(K), Device::gpu),
              dtype,
              Device::gpu);
          // Each row alone is the one-row kernel.
          std::vector<array> alone;
          for (int i = 0; i < limit; ++i) {
            alone.push_back(product(q, rows(x, i, i + 1)));
          }
          auto singles = concatenate(alone, 0, Device::gpu);
          eval(singles);
          // From 9 rows the product has two or more tiles of rows.
          for (int m = 2; m <= limit; ++m) {
            CAPTURE(m);
            CHECK(same_bytes(product(q, rows(x, 0, m)), rows(singles, 0, m)));
          }
          CHECK(same_bytes(
              product(q, rows(x, limit - 2, limit)),
              rows(singles, limit - 2, limit)));
          // Leading dimensions of x are more rows of one product.
          if (limit >= 6) {
            auto y = product(q, reshape(rows(x, 0, 6), {2, 3, K}, Device::gpu));
            CHECK(same_bytes(
                reshape(y, {6, N}, Device::gpu), rows(singles, 0, 6)));
          }
        }
      }
    }
  }
}

TEST_CASE("quantized matmul broadcast batches keep the bits of one row") {
  // A batch of one-row products with one weight matrix (x [..., 1, K], the
  // weights broadcast over the batch). With MLX_QMV_ROWS_BROADCAST on (the
  // default) the row-exact kernel serves it; the qmv_rows_broadcast_off ctest
  // entry runs this case with the batched qmv (tests/CMakeLists.txt). Each row
  // must have the bytes of the row alone either way.
  const char* value = std::getenv("MLX_QMV_ROWS_BROADCAST");
  bool on = value == nullptr || std::atoi(value) != 0;
  REQUIRE(env::qmv_rows_broadcast() == on);

  // K selects the one-row kernel: fast (K a multiple of 512 for 2 and 4 bits,
  // of 256 for 8 bits), general (with a short last block), or qmv_quad
  // (K = 64 or 128). 2 bits outside the fast kernel keep the batched qmv.
  std::vector<std::pair<int, int>> shapes = {
      {2048, 1024}, {61, 5120}, {1, 2048}, {1000, 768}, {5, 320}, {9, 128}};
  for (auto [N, K] : shapes) {
    for (int bits : {2, 4, 8}) {
      for (int group_size : {32, 64, 128}) {
        if (K % group_size != 0) {
          continue;
        }
        for (auto dtype : {float32, float16, bfloat16}) {
          CAPTURE(N);
          CAPTURE(K);
          CAPTURE(bits);
          CAPTURE(group_size);
          CAPTURE(dtype);
          auto q = weights(N, K, group_size, bits, dtype, N + K + bits + 7);
          Weights b{
              group_size,
              bits,
              expand_dims(q.w, 0, Device::gpu),
              expand_dims(q.scales, 0, Device::gpu),
              expand_dims(q.biases, 0, Device::gpu)};
          constexpr int count = 19;
          auto x = astype(
              random::normal(
                  {count, K},
                  float32,
                  0.0f,
                  1.0f,
                  random::key(K + 1),
                  Device::gpu),
              dtype,
              Device::gpu);
          std::vector<array> alone;
          for (int i = 0; i < count; ++i) {
            alone.push_back(product(q, rows(x, i, i + 1)));
          }
          auto singles = concatenate(alone, 0, Device::gpu);
          eval(singles);
          // 2 to 19 rows: one tile, two tiles and a partial last tile.
          for (int m : {2, 3, 7, 8, 9, 16, 19}) {
            CAPTURE(m);
            auto y = product(b, reshape(rows(x, 0, m), {m, 1, K}, Device::gpu));
            CHECK(same_bytes(
                reshape(y, {m, N}, Device::gpu), rows(singles, 0, m)));
          }
          // Two batch dimensions are more rows of one batch.
          auto y =
              product(b, reshape(rows(x, 0, 6), {2, 3, 1, K}, Device::gpu));
          CHECK(
              same_bytes(reshape(y, {6, N}, Device::gpu), rows(singles, 0, 6)));
        }
      }
    }
  }
}
