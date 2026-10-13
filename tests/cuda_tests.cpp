// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

#ifdef _WIN32
#include "mlx/backend/cuda/wddm.h"
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

using namespace mlx::core;

namespace {

bool memory_pools_supported() {
  int supported = 0;
  REQUIRE_EQ(
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0),
      cudaSuccess);
  return supported;
}

struct CacheLimitGuard {
  ~CacheLimitGuard() {
    set_cache_limit(previous);
  }

  size_t previous;
};

struct MemoryLimitGuard {
  ~MemoryLimitGuard() {
    set_memory_limit(previous);
  }

  size_t previous;
};

} // namespace

constexpr uint64_t gib = 1ULL << 30;

#ifdef _WIN32
TEST_CASE("test memory limit from budget") {
  CHECK_EQ(
      cu::memory_limit_from_budget(24 * gib, 8 * gib, 6 * gib),
      22 * gib - (24 * gib) / 20);
  CHECK_EQ(
      cu::memory_limit_from_budget(24 * gib, 4 * gib, 6 * gib),
      24 * gib - (24 * gib) / 20);
  CHECK_EQ(cu::memory_limit_from_budget(24 * gib, 25 * gib, 0), 0);
}

TEST_CASE("test wddm memory limit") {
  MemoryLimitGuard memory_limit{
      set_memory_limit(std::numeric_limits<size_t>::max())};
  cudaDeviceProp default_properties{};
  REQUIRE_EQ(cudaGetDeviceProperties(&default_properties, 0), cudaSuccess);
  auto default_limit = get_memory_limit();
  if (default_properties.memoryPoolsSupported &&
      !default_properties.integrated && !default_properties.tccDriver &&
      default_properties.luidDeviceNodeMask != 0) {
    CHECK_LT(default_limit, std::numeric_limits<size_t>::max());
  }
}
#endif // _WIN32

TEST_CASE("test clear cache trims CUDA pool") {
  if (!memory_pools_supported()) {
    return;
  }

  cudaMemPool_t pool = nullptr;
  REQUIRE_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);

  CacheLimitGuard cache_limit{set_cache_limit(gib)};
  clear_cache();

  uint64_t initial_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &initial_reserved),
      cudaSuccess);

  auto s = default_stream(Device::gpu);
  {
    auto a = zeros({16 * 1024 * 1024}, float32, s);
    eval(a);
    synchronize(s);
  }
  CHECK_GE(get_cache_memory(), 64ULL << 20);

  uint64_t allocated_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &allocated_reserved),
      cudaSuccess);
  CHECK_GT(allocated_reserved, initial_reserved);

  clear_cache();
  uint64_t final_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &final_reserved),
      cudaSuccess);
  CHECK_LT(final_reserved, allocated_reserved);
}

TEST_CASE("test fused sdpa at head dims 256 and 512") {
  // 1..3 query rows take the vector kernel (one and two pass), prefill-sized
  // blocks at head dim 256 take cuDNN on sm_120; each must match an fp32
  // reference built from plain ops.
  auto reference = [](array q, array k, array v, float scale, bool causal) {
    int gqa = q.shape(1) / k.shape(1);
    int qL = q.shape(2), kL = k.shape(2);
    k = repeat(astype(k, float32), gqa, 1);
    v = repeat(astype(v, float32), gqa, 1);
    auto s = matmul(astype(q, float32) * scale, swapaxes(k, -1, -2));
    if (causal) {
      auto rows = reshape(arange(qL) + (kL - qL), {qL, 1});
      auto cols = reshape(arange(kL), {1, kL});
      s = where(rows >= cols, s, array(-INFINITY));
    }
    return matmul(softmax(s, -1), v);
  };
  int cc_major = 0;
  REQUIRE_EQ(
      cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0),
      cudaSuccess);
  for (int D : {256, 512}) {
    for (int qL : {1, 3, 64}) {
      if (qL == 64 && (D != 256 || cc_major < 12)) {
        continue;
      }
      for (int kL : {qL == 64 ? 64 : 37, 1500}) {
        for (bool causal : {false, true}) {
          auto q = random::normal({1, 8, qL, D}, bfloat16);
          // Keys and values as views into a longer cache, like a decode.
          auto k = slice(
              random::normal({1, 2, kL + 16, D}, bfloat16),
              {0, 0, 0, 0},
              {1, 2, kL, D});
          auto v = slice(
              random::normal({1, 2, kL + 16, D}, bfloat16),
              {0, 0, 0, 0},
              {1, 2, kL, D});
          float scale = 1.0f / std::sqrt(static_cast<float>(D));
          auto out = fast::scaled_dot_product_attention(
              q, k, v, scale, causal ? "causal" : "", {}, {}, true);
          auto expected = reference(q, k, v, scale, causal);
          CHECK(allclose(astype(out, float32), expected, 2e-2, 2e-2)
                    .item<bool>());
        }
      }
    }
  }
}

TEST_CASE("test quantized matmul over a few rows") {
  // Rows 2..7 run the tiled kernel (one weight read for all rows) where the
  // GPU has it; each row must match a dequantized reference, batched x too.
  for (int bits : {4, 8}) {
    auto w = random::normal({384, 512}, bfloat16);
    auto q = quantize(w, 64, bits);
    auto w_dq = astype(dequantize(q[0], q[1], q[2], 64, bits), float32);
    for (Shape x_shape :
         {Shape{1, 512},
          Shape{2, 512},
          Shape{5, 512},
          Shape{7, 512},
          Shape{2, 3, 512}}) {
      auto x = random::normal(x_shape, bfloat16);
      auto out = quantized_matmul(x, q[0], q[1], q[2], true, 64, bits);
      auto expected = matmul(astype(x, float32), transpose(w_dq));
      CHECK(allclose(astype(out, float32), expected, 2e-2, 2e-1).item<bool>());
    }
  }
}
