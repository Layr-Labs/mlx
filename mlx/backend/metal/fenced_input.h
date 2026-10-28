// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

// The wait of a custom kernel on a fenced input from a CPU stream, given to
// the kernel by Fence::wait_in_consumer instead of a fence_wait dispatch.
struct FencedInput {
  const Primitive* consumer;
  int input;
  std::shared_ptr<void> fence;
  uint32_t count;
};

// Remove the waits given to `consumer` and return them. Discard all others.
std::vector<FencedInput> take_fenced_inputs(const Primitive* consumer);

// Bind the timestamp buffer at `index` and the count at `index + 1`.
void bind_fenced_input(CommandEncoder& enc, const FencedInput& f, int index);

// Encode the wait as Fence::wait does, for an input read before the kernel.
void encode_fence_wait(Stream stream, const array& x, const FencedInput& f);

} // namespace mlx::core::metal
