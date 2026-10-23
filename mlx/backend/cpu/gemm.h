// Copyright © 2025 Apple Inc.

#pragma once
#include <algorithm>

#include "mlx/array.h"

namespace mlx::core {

// A BLAS gemm with beta = 0 must not read C, but Accelerate can (seen with
// M = 1). A buffer from the allocator cache can hold NaN, and NaN * 0 is NaN.
// Call this on the C region of every BLAS gemm with beta = 0.
template <typename T>
inline void clear_gemm_output(T* c, size_t rows, size_t cols, size_t ldc) {
  for (size_t r = 0; r < rows; ++r) {
    std::fill_n(c + r * ldc, cols, T(0));
  }
}

template <typename T>
void matmul(
    const T* a,
    const T* b,
    T* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides);

} // namespace mlx::core
