// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/api.h"

#if defined(__APPLE__)
#ifdef __cplusplus
extern "C" {
#endif

/// Commit limits for the Metal command buffers that the calling thread
/// encodes: a buffer is committed once it holds more than `max_ops`
/// operations or more than `max_mb` megabytes of distinct inputs. A value of
/// zero keeps the device default for that limit. Stores the thread's
/// previous values (zero for the default) in `previous_ops` and `previous_mb`
/// when they are not null. Partitioning work into command buffers does not
/// change results; larger buffers let the CPU encode further ahead of the
/// GPU and keep each buffer's temporaries alive until it completes.
MLX_API void mlx_metal_set_thread_command_buffer_limits(
    int max_ops,
    int max_mb,
    int* previous_ops,
    int* previous_mb);

/// Output tile (32, 64, 6432, 3264 or 6464: rows x columns[, K depth]) of the
/// steel affine quantized matmul for the
/// products the calling thread encodes on a GPU without neural accelerators;
/// any other value restores MLX_QMM_STEEL_TILE. Stores the thread's previous
/// override (0 for none) in `previous_tile` when it is not null. The tile
/// changes which threadgroup computes an element, never its bits. Builds whose
/// kernels come only from the precompiled library (MLX_METAL_JIT=OFF) keep
/// 32 x 32 whatever the tile.
MLX_API void mlx_metal_set_thread_qmm_steel_tile(int tile, int* previous_tile);

#ifdef __cplusplus
}
#endif
#endif
