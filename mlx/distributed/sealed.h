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
using SealFn = int (*)(
    void* ctx,
    const void* plaintext,
    size_t n,
    void* frame,
    uint64_t counter);
using OpenFn = int (*)(
    void* ctx,
    const void* frame,
    size_t n,
    void* plaintext,
    uint64_t counter,
    int peer);

// The streamed frame: piece k holds plaintext
// [k * piece_bytes, min(n, (k + 1) * piece_bytes)) as its ciphertext, then a
// kSealTagBytes tag; the pieces are in order. seal_piece writes
// len + kSealTagBytes bytes; open_piece reads them and writes len bytes.
// Calls for one counter can run at the same time, on any thread, in any
// order; `pieces` and `total` (n) are the same for each piece of a counter.
// Both return 0 on success.
constexpr size_t kSealTagBytes = 16;
using SealPieceFn = int (*)(
    void* ctx,
    const void* plaintext,
    size_t len,
    void* sealed,
    uint64_t counter,
    uint32_t piece,
    uint32_t pieces,
    size_t total);
using OpenPieceFn = int (*)(
    void* ctx,
    const void* sealed,
    size_t len,
    void* plaintext,
    uint64_t counter,
    uint32_t piece,
    uint32_t pieces,
    size_t total,
    int peer);

struct SealCodec {
  SealFn seal = nullptr;
  OpenFn open = nullptr;
  void* ctx = nullptr;
  size_t overhead = 0;
  // Optional. A group that can exchange (two-rank JACCL mesh) then streams
  // the pieces; other groups use seal and open.
  SealPieceFn seal_piece = nullptr;
  OpenPieceFn open_piece = nullptr;
  size_t piece_bytes = 0;
  bool streamed() const {
    return seal_piece != nullptr && open_piece != nullptr;
  }
};

// Installs (or, with a null seal, removes) the process's codec. A sealed
// all-sum without a codec is refused when the graph is built. A streamed
// codec needs piece_bytes: a multiple of 16 in [4 KiB, 16 MiB].
MLX_API void set_seal_codec(SealCodec codec);
MLX_API std::optional<SealCodec> seal_codec();

// The sum of two ranks' arrays, crossing the link sealed. Two ranks only;
// float32, bfloat16 and float16; the sum is taken in rank order (rank 0 +
// rank 1) so both ranks hold the same bits.
MLX_API array all_sum_sealed(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

// Two ranks, `x` shaped [1, T, ...]. Rank r receives rows
// [r * T / 2, (r + 1) * T / 2) of the sum, added in rank order exactly as
// all_sum_sealed adds them (T even): each rank sends only the peer's half.
MLX_API array reduce_scatter_sealed(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

// Two ranks, `x` shaped [1, T, ...] (this rank's rows): [1, 2T, ...], rank
// 0's rows then rank 1's, each rank sending only its own. A copy: every
// element keeps its bits.
MLX_API array all_gather_sealed(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::distributed
