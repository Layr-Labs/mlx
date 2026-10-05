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

// D512-2PASS. The 2-pass split-K kernels ONLY, with no single-pass twin.
//
// `sdpa_vector` holds q, k and o in registers at D / 32 floats each and is
// launched at the Metal maximum 1024 threads per threadgroup
// (scaled_dot_product_attention.cpp:461). At D = 512 that is 48 live floats
// per thread against 1024 threads, an occupancy claim the 2-pass kernel does
// not make (it runs 32 x gqa_factor threads with 32 live floats), and the
// host routes every D = 512 call to the 2-pass form for exactly that reason.
// Instantiating the single-pass twin would only add a pipeline nothing can
// dispatch.
#define instantiate_sdpa_vector_2pass(type, qk_dim, value_dim)  \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
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
