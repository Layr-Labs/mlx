// Copyright © 2026 Darkbloom.
//
// A sealed all-sum: the two ranks' partials cross the link only as
// authenticated, encrypted frames made and opened by a codec the embedding
// program registers (Darkbloom: AES-256-GCM with directional keys from its
// keyed data plane). Evaluated on the CPU stream inside the lazy graph, like
// the native all-sum, so the model thread never synchronizes the GPU for it.
// A frame that does not open is refused: the codec reports it and the
// evaluation fails; there is no plaintext fallback.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "mlx/api.h"
#include "mlx/distributed/distributed.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

// seal: plaintext (n bytes) -> frame (n + overhead bytes) for `counter`;
// open: the peer's frame -> plaintext (n bytes) for `counter` from `peer`.
// Both return 0 on success. `counter` advances by one per sealed all-sum, in
// evaluation order, identically on both ranks.
using SealFn = int (*)(void* ctx, const void* plaintext, size_t n, void* frame, uint64_t counter);
using OpenFn = int (*)(void* ctx, const void* frame, size_t n, void* plaintext, uint64_t counter, int peer);

struct SealCodec {
  SealFn seal = nullptr;
  OpenFn open = nullptr;
  void* ctx = nullptr;
  size_t overhead = 0;
};

// Installs (or, with a null seal, removes) the process's codec. A sealed
// all-sum without a codec is refused when the graph is built.
MLX_API void set_seal_codec(SealCodec codec);
MLX_API std::optional<SealCodec> seal_codec();

// The sum of two ranks' arrays, crossing the link sealed. Two ranks only;
// float32, bfloat16 and float16; the sum is taken in rank order (rank 0 +
// rank 1) so both ranks hold the same bits.
MLX_API array all_sum_sealed(const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});

} // namespace mlx::core::distributed
