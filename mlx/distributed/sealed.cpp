// Copyright © 2026 Darkbloom.

#include <algorithm>
#include <dispatch/dispatch.h>
#include <mutex>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/primitives.h"
#include "mlx/distributed/sealed.h"

namespace mlx::core::distributed {

namespace {
std::mutex codec_mutex;
std::optional<SealCodec> installed_codec;
} // namespace

void set_seal_codec(SealCodec codec) {
  std::lock_guard<std::mutex> lock(codec_mutex);
  if (codec.seal == nullptr || codec.open == nullptr) {
    installed_codec.reset();
  } else {
    installed_codec = codec;
  }
}

std::optional<SealCodec> seal_codec() {
  std::lock_guard<std::mutex> lock(codec_mutex);
  return installed_codec;
}

array all_sum_sealed(
    const array& x,
    std::optional<Group> group_,
    StreamOrDevice s) {
  auto group = group_.has_value() ? group_.value() : distributed::init();
  if (group.size() != 2) {
    throw std::invalid_argument("[all_sum_sealed] needs exactly two ranks");
  }
  if (x.dtype() != float32 && x.dtype() != bfloat16 && x.dtype() != float16) {
    throw std::invalid_argument(
        "[all_sum_sealed] sums float32, bfloat16 or float16 partials only");
  }
  auto codec = seal_codec();
  if (!codec.has_value()) {
    throw std::runtime_error(
        "[all_sum_sealed] no seal codec is installed: a partial never crosses the link in the clear");
  }
  auto stream = detail::communication_stream(group, s);
  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<SealedAllSum>(stream, group, *codec),
      {x});
}

namespace {
template <typename T>
void add_span(const T* rank0, const T* rank1, T* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = static_cast<T>(
        static_cast<float>(rank0[i]) + static_cast<float>(rank1[i]));
  }
}
template <>
void add_span<float>(
    const float* rank0,
    const float* rank1,
    float* out,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = rank0[i] + rank1[i];
  }
}
// The same element-wise sum (one rounding, rank 0 + rank 1), split across
// cores for prefill-sized partials; small partials stay on this thread.
template <typename T>
void add_in_rank_order(const T* rank0, const T* rank1, T* out, size_t n) {
  constexpr size_t kSpan = size_t(1) << 18;
  if (n <= kSpan) {
    add_span(rank0, rank1, out, n);
    return;
  }
  size_t spans = (n + kSpan - 1) / kSpan;
  dispatch_apply(spans, DISPATCH_APPLY_AUTO, ^(size_t i) {
    size_t begin = i * kSpan;
    size_t count = std::min(kSpan, n - begin);
    add_span(rank0 + begin, rank1 + begin, out + begin, count);
  });
}
} // namespace

void SealedAllSum::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto in = inputs[0].flags().row_contiguous
      ? inputs[0]
      : contiguous_copy_cpu(inputs[0], stream());
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  size_t n = in.nbytes();
  size_t frame = n + codec_.overhead;
  array sealed({static_cast<int>(frame)}, uint8, nullptr, {});
  sealed.set_data(allocator::malloc(frame));
  array gathered({static_cast<int>(2 * frame)}, uint8, nullptr, {});
  gathered.set_data(allocator::malloc(2 * frame));
  array peer(in.shape(), in.dtype(), nullptr, {});
  peer.set_data(allocator::malloc(n));

  int rank = group().rank();
  uint64_t counter = counter_++;
  auto codec = codec_;
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(sealed);
  encoder.dispatch(
      [codec, src = in.data<char>(), dst = sealed.data<char>(), n, counter]() {
        if (codec.seal(codec.ctx, src, n, dst, counter) != 0) {
          throw std::runtime_error(
              "[all_sum_sealed] the codec refused to seal a partial");
        }
      });
  distributed::detail::all_gather(group(), sealed, gathered, stream());
  encoder.set_input_array(gathered);
  encoder.set_output_array(out);
  encoder.dispatch([codec,
                    frames = gathered.data<char>(),
                    mine = in.data<char>(),
                    theirs = peer.data<char>(),
                    dst = out.data<char>(),
                    n,
                    frame,
                    counter,
                    rank,
                    dtype = in.dtype()]() {
    const char* peer_frame = frames + (rank == 0 ? frame : 0);
    if (codec.open(codec.ctx, peer_frame, n, theirs, counter, 1 - rank) != 0) {
      throw std::runtime_error(
          "[all_sum_sealed] a peer frame did not authenticate: refused");
    }
    const char* r0 = rank == 0 ? mine : theirs;
    const char* r1 = rank == 0 ? theirs : mine;
    switch (dtype) {
      case float32:
        add_in_rank_order(
            reinterpret_cast<const float*>(r0),
            reinterpret_cast<const float*>(r1),
            reinterpret_cast<float*>(dst),
            n / sizeof(float));
        break;
      case bfloat16:
        add_in_rank_order(
            reinterpret_cast<const bfloat16_t*>(r0),
            reinterpret_cast<const bfloat16_t*>(r1),
            reinterpret_cast<bfloat16_t*>(dst),
            n / sizeof(bfloat16_t));
        break;
      case float16:
        add_in_rank_order(
            reinterpret_cast<const float16_t*>(r0),
            reinterpret_cast<const float16_t*>(r1),
            reinterpret_cast<float16_t*>(dst),
            n / sizeof(float16_t));
        break;
      default:
        throw std::runtime_error("[all_sum_sealed] unsupported dtype");
    }
  });
  encoder.add_temporary(sealed);
  encoder.add_temporary(gathered);
  encoder.add_temporary(peer);
  if (!inputs[0].flags().row_contiguous) {
    encoder.add_temporary(in);
  }
}

void SealedAllSum::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[SealedAllSum::eval_gpu] has no GPU implementation.");
}

} // namespace mlx::core::distributed
