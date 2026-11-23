// Copyright © 2026 Eigen Labs.
#include <cmath>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/backend/common/command_buffer_limits.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

namespace {

constexpr int reference_rows = 4096;

// The calling thread's prefill reference rows for the scope.
struct ScopedPrefillRows {
  int previous = 0;
  explicit ScopedPrefillRows(int rows) {
    mlx_metal_set_thread_prefill_rows(rows, &previous);
  }
  ~ScopedPrefillRows() {
    mlx_metal_set_thread_prefill_rows(previous, nullptr);
  }
};

// True when a and b have the same shape and the same bytes.
bool same_bits(const array& a, const array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
    return false;
  }
  auto bits = a.itemsize() == 2 ? uint16 : uint32;
  return array_equal(view(a, bits, Device::cpu), view(b, bits, Device::cpu))
      .item<bool>();
}

array rows_of(const array& a, int start, int stop) {
  return contiguous(
      slice(a, {start, 0}, {stop, a.shape(1)}, Device::gpu),
      false,
      Device::gpu);
}

struct Product {
  array x;
  std::vector<array> q;

  // 4-bit affine weights in groups of 64, as the Qwen 3.5 checkpoints use.
  Product(int M, int N, int K, Dtype dtype)
      : x(astype(
            random::normal({M, K}, float32, 0.0f, 1.0f, random::key(K + N)),
            dtype,
            Device::gpu)),
        q(quantize(
            astype(
                random::normal(
                    {N, K},
                    float32,
                    0.0f,
                    1.0f / std::sqrt(float(K)),
                    random::key(3 * K + N)),
                dtype,
                Device::gpu),
            64,
            4,
            "affine",
            std::nullopt,
            Device::gpu)) {}

  array operator()(const array& rows) const {
    return quantized_matmul(
        rows, q[0], q[1], q[2], true, 64, 4, "affine", Device::gpu);
  }
};

} // namespace

TEST_CASE("prefill rows scope changes only the ops built inside it") {
  CHECK_EQ(mlx_metal_thread_prefill_rows(), 0);
  Product product(256, 1024, 4096, bfloat16);

  // Evaluation detaches an array's primitive, so keep the primitives of the
  // arrays whose routes are compared.
  auto before = product(product.x);
  auto before_p = before.primitive_ptr();
  eval(before);

  array inside_built(0.0f);
  array inside_evaluated(0.0f);
  std::shared_ptr<Primitive> inside_p;
  {
    ScopedPrefillRows scope(reference_rows);
    CHECK_EQ(mlx_metal_thread_prefill_rows(), reference_rows);
    inside_built = product(product.x);
    inside_p = inside_built.primitive_ptr();
    inside_evaluated = product(product.x);
    eval(inside_evaluated);
  }
  CHECK_EQ(mlx_metal_thread_prefill_rows(), 0);

  // Built outside a scope, evaluated inside one.
  auto outside_built = product(product.x);
  {
    ScopedPrefillRows scope(reference_rows);
    eval(outside_built);
  }
  auto after = product(product.x);
  auto after_p = after.primitive_ptr();
  eval(after);
  // Built inside the scope, evaluated after it ended.
  eval(inside_built);

  CHECK_EQ(static_cast<QuantizedMatmul&>(*before_p).prefill_rows(), 0);
  CHECK_EQ(
      static_cast<QuantizedMatmul&>(*inside_p).prefill_rows(), reference_rows);
  CHECK(before_p->is_equivalent(*after_p));
  CHECK_FALSE(before_p->is_equivalent(*inside_p));

  // Ops built outside the scope have the bits of ops that never saw one,
  // whatever the thread's scope when they are evaluated.
  CHECK(same_bits(after, before));
  CHECK(same_bits(outside_built, before));
  // An op built in the scope keeps its route after the scope ends.
  CHECK(same_bits(inside_built, inside_evaluated));
}

TEST_CASE("short products in the prefill rows scope match the reference rows") {
  // The rows of a product built in the scope have the bits of the same rows
  // in a product of `reference_rows` rows, at every row count from 2. The
  // shapes are the Qwen 3.5 9B projections whose split-K choice depends on
  // the row count outside the scope (the K/V projections, N = 1024).
  struct Shape {
    int N;
    int K;
  };
  for (auto [N, K] :
       {Shape{1024, 4096}, Shape{4096, 12288}, Shape{12288, 4096}}) {
    for (auto dtype : {bfloat16, float16}) {
      CAPTURE(N);
      CAPTURE(K);
      CAPTURE(dtype);
      Product product(reference_rows, N, K, dtype);
      ScopedPrefillRows scope(reference_rows);
      auto reference = product(product.x);
      eval(reference);
      int outside_differs = 0;
      for (int M : {2,   3,   7,   16,  31,  33,  64,  65,   100,  128,
                    129, 200, 256, 257, 384, 511, 512, 1000, 1024, 2048}) {
        CAPTURE(M);
        auto rows = rows_of(product.x, 0, M);
        auto inside = product(rows);
        eval(inside);
        CHECK(same_bits(inside, rows_of(reference, 0, M)));
        {
          ScopedPrefillRows none(0);
          auto outside = product(rows);
          eval(outside);
          outside_differs += !same_bits(outside, rows_of(reference, 0, M));
        }
      }
      // How many row counts take another route without the scope on this
      // GPU; informational, it depends on the device.
      MESSAGE(
          "row counts with other bits outside the scope: ", outside_differs);
    }
  }
}

TEST_CASE("compiled functions trace again inside the prefill rows scope") {
  Product product(256, 1024, 4096, bfloat16);
  auto compiled = compile([](const std::vector<array>& inputs) {
    return std::vector<array>{quantized_matmul(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        true,
        64,
        4,
        "affine",
        Device::gpu)};
  });
  std::vector<array> inputs{
      product.x, product.q[0], product.q[1], product.q[2]};

  auto outside = compiled(inputs)[0];
  eval(outside);
  array inside(0.0f);
  array inside_direct(0.0f);
  {
    ScopedPrefillRows scope(reference_rows);
    inside = compiled(inputs)[0];
    inside_direct = product(product.x);
    eval(inside, inside_direct);
  }
  auto outside_again = compiled(inputs)[0];
  auto outside_direct = product(product.x);
  eval(outside_again, outside_direct);

  CHECK(same_bits(inside, inside_direct));
  CHECK(same_bits(outside, outside_direct));
  CHECK(same_bits(outside_again, outside_direct));
}

TEST_CASE("head-dim 256 attention in the prefill rows scope") {
  // Qwen 3.5 9B full attention: 16 query heads, 4 key/value heads, head dim
  // 256. The last 128 queries of a causal 2048-query call, attended alone
  // over the same keys, as a prompt chunk that starts at 1920.
  constexpr int L = 2048, tail = 128, H = 16, Hkv = 4, D = 256;
  auto normal = [](Shape shape, int seed) {
    return astype(
        random::normal(shape, float32, 0.0f, 1.0f, random::key(seed)),
        bfloat16,
        Device::gpu);
  };
  auto q = normal({1, H, L, D}, 1);
  auto k = normal({1, Hkv, L, D}, 2);
  auto v = normal({1, Hkv, L, D}, 3);
  auto q_tail = slice(q, {0, 0, L - tail, 0}, {1, H, L, D}, Device::gpu);
  float scale = 1.0f / std::sqrt(float(D));
  auto attend = [&](const array& queries) {
    return fast::scaled_dot_product_attention(
        queries,
        k,
        v,
        scale,
        "causal",
        std::nullopt,
        std::nullopt,
        false,
        Device::gpu);
  };

  auto outside = attend(q_tail);
  array inside(0.0f);
  array full(0.0f);
  {
    ScopedPrefillRows scope(reference_rows);
    inside = attend(q_tail);
    full = attend(q);
  }
  eval(outside, inside, full);
  if (metal::is_nax_available()) {
    // Both calls take the fused NAX kernel in the scope, so the chunk's
    // queries have the bits of the same queries in the longer call.
    auto full_tail =
        slice(full, {0, 0, L - tail, 0}, {1, H, L, D}, Device::gpu);
    CHECK(same_bits(inside, full_tail));
  } else {
    // Without neural accelerators the scope does not change attention.
    CHECK(same_bits(inside, outside));
  }
}
