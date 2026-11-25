// Copyright © 2026 Eigen Labs.
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

// The ctest entry qmv_tile runs this case again with MLX_QMV_TILE=1, so that
// each GPU without neural accelerators uses the split-K tile kernel.
TEST_CASE("quantized matmul of 6 to 16 rows matches the dequantized product") {
  struct Shape {
    int K;
    int N;
  };
  // One K part, two even K parts, and 8 parts for 9 chunks of 256.
  std::vector<Shape> shapes = {{256, 1024}, {512, 256}, {2304, 4096}};
  uint64_t seed = 0;
  for (auto dtype : {float16, bfloat16}) {
    float tol = dtype == float16 ? 2e-3f : 1e-2f;
    for (int group_size : {32, 64, 128}) {
      for (auto [K, N] : shapes) {
        auto w =
            random::normal({N, K}, float32, 0.0f, 0.05f, random::key(seed++));
        auto q = quantize(w, group_size, 4, "affine");
        auto wq = q[0];
        auto scales = astype(q[1], dtype);
        auto biases = astype(q[2], dtype);
        auto w_hat = dequantize(
            wq,
            astype(scales, float32),
            astype(biases, float32),
            group_size,
            4,
            "affine");
        for (int M : {6, 7, 8, 9, 13, 16}) {
          auto x = astype(
              random::normal({M, K}, float32, 0.0f, 2.0f, random::key(seed++)),
              dtype);
          auto y = quantized_matmul(
              x, wq, scales, biases, true, group_size, 4, "affine");
          auto y_hat = matmul(astype(x, float32), transpose(w_hat));
          auto err =
              max(abs(subtract(astype(y, float32), y_hat))).item<float>();
          auto scale = max(abs(y_hat)).item<float>();
          INFO(
              "dtype ",
              dtype,
              " group_size ",
              group_size,
              " K ",
              K,
              " N ",
              N,
              " M ",
              M);
          CHECK_EQ(y.dtype(), dtype);
          CHECK_LT(err, tol * scale);
        }
      }
    }
  }
}
