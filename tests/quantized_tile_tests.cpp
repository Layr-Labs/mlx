// Copyright © 2026 Eigen Labs.
#include <cmath>

#include "doctest/doctest.h"
#include "mlx/backend/common/command_buffer_limits.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
// The calling thread's steel quantized-matmul tile for the scope; 0 is the
// default policy.
struct ScopedSteelTile {
  int previous = 0;
  explicit ScopedSteelTile(int tile) {
    mlx_metal_set_thread_qmm_steel_tile(tile, &previous);
  }
  ~ScopedSteelTile() {
    mlx_metal_set_thread_qmm_steel_tile(previous, nullptr);
  }
};
} // namespace

TEST_CASE("prompt-sized affine qmm in every kernel build") {
  // 512 rows and 1024 outputs take a tile wider than 32 x 32 by default where
  // kernels are compiled at run time. A build whose kernels come only from the
  // precompiled library must keep 32 x 32 rather than ask that library for a
  // tile it never instantiated. The product fills 512 threadgroups at 32 x 32,
  // so it runs as one qmm rather than split along K. K = 96 keeps the steel
  // kernel on GPUs with neural accelerators too (theirs needs K % 64 == 0);
  // K = 192 in groups of 64 adds the 64-deep tile where none serves it.
  constexpr int M = 512, N = 1024, bits = 4;
  for (auto [K, group_size] : {std::pair{96, 32}, std::pair{192, 64}}) {
    auto x = reshape(
        divide(
            sin(arange(M * K, float32, Device::gpu) * array(0.007f),
                Device::gpu),
            array(std::sqrt(float(K))),
            Device::gpu),
        {M, K});
    auto w = reshape(
        divide(
            cos(arange(N * K, float32, Device::gpu) * array(0.013f),
                Device::gpu),
            array(std::sqrt(float(K))),
            Device::gpu),
        {N, K});
    for (auto dtype : {float16, bfloat16, float32}) {
      CAPTURE(K);
      CAPTURE(dtype);
      auto xt = astype(x, dtype, Device::gpu);
      auto q = quantize(
          astype(w, dtype, Device::gpu),
          group_size,
          bits,
          "affine",
          std::nullopt,
          Device::gpu);
      auto product = [&]() {
        auto out = quantized_matmul(
            xt,
            q[0],
            q[1],
            q[2],
            true,
            group_size,
            bits,
            "affine",
            Device::gpu);
        eval(out);
        return out;
      };
      array reference(0.0f);
      {
        ScopedSteelTile tile(32);
        reference = product();
      }
      auto w_hat = astype(
          dequantize(
              q[0],
              q[1],
              q[2],
              group_size,
              bits,
              "affine",
              {},
              {},
              Device::cpu),
          float32,
          Device::cpu);
      auto expected = matmul(
          astype(xt, float32, Device::cpu),
          transpose(w_hat, Device::cpu),
          Device::cpu);
      CHECK(allclose(
                astype(reference, float32, Device::cpu),
                expected,
                1e-2,
                1e-2,
                false,
                Device::cpu)
                .item<bool>());
      // Every tile computes each element from the same K slabs in the same
      // order, so each gives the 32 x 32 bytes.
      for (int t : {0, 64, 6432, 3264, 6464}) {
        CAPTURE(t);
        ScopedSteelTile tile(t);
        CHECK(array_equal(product(), reference, Device::cpu).item<bool>());
      }
    }
  }
}
