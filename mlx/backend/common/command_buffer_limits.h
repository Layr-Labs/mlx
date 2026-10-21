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

#ifdef __cplusplus
}
#endif
#endif
