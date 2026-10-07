#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_fp32partials_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_fp32partials_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

// `split` is the merge-plane publish count, NOT part of the kernel name --
// the host names this kernel by type and dims only. SPLIT = 1 is the shipped
// body instruction for instruction; a larger value only shrinks the
// threadgroup allocation (see sdpa_vector.h).
#define instantiate_sdpa_vector_gqa(type, qk_dim, value_dim, hpt, split) \
  instantiate_kernel(                                              \
      "sdpa_vector_2pass_fp32partials_1_gqa_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1_gqa,                                     \
      type,                                                        \
      qk_dim,                                                      \
      value_dim,                                                   \
      8,                                                           \
      hpt,                                                         \
      split)

// D512-2PASS. 2-pass kernels only. The host never runs `sdpa_vector` at
// D = 512, because at 1024 threads it would hold 48 floats per thread.
#define instantiate_sdpa_vector_2pass(type, qk_dim, value_dim)  \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_fp32partials_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 192, 128)        \
  instantiate_sdpa_vector(type, 192, 192)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_2pass(type, 512, 512)  \
  instantiate_sdpa_vector_gqa(type, 64, 64, 8, 1)      \
  instantiate_sdpa_vector_gqa(type, 128, 128, 4, 1)   \
  instantiate_sdpa_vector_gqa(type, 512, 512, 2, 2)   \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 192) \
  instantiate_sdpa_vector_aggregation(type, 256) \
  instantiate_sdpa_vector_aggregation(type, 512)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
