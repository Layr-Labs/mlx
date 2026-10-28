// Copyright © 2026 Darkbloom.

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/primitives.h"
#include "mlx/distributed/sealed.h"
#include "mlx/distributed/sealed_stream.h"

namespace mlx::core::distributed {

namespace {
std::mutex codec_mutex;
std::optional<SealCodec> installed_codec;
// One counter for every sealed collective, advanced per evaluation; both
// ranks evaluate the same graph in the same order.
std::atomic<uint64_t> sealed_counter{0};
uint64_t next_sealed_counter() {
  return sealed_counter++;
}
} // namespace

void set_seal_codec(SealCodec codec) {
  if (codec.seal != nullptr && codec.open != nullptr &&
      (codec.seal_piece != nullptr || codec.open_piece != nullptr)) {
    if (!codec.streamed() || codec.piece_bytes < 4096 ||
        codec.piece_bytes > (size_t(16) << 20) || codec.piece_bytes % 16 != 0) {
      throw std::invalid_argument(
          "[set_seal_codec] a streamed codec needs both piece callbacks and "
          "piece_bytes: a multiple of 16 in [4 KiB, 16 MiB]");
    }
  }
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
// [1, T, ...] split on its second axis between two ranks.
void require_rows(const array& x, Group& group, const char* name, bool even) {
  if (group.size() != 2) {
    throw std::invalid_argument(std::string(name) + " needs exactly two ranks");
  }
  if (x.dtype() != float32 && x.dtype() != bfloat16 && x.dtype() != float16) {
    throw std::invalid_argument(
        std::string(name) + " moves float32, bfloat16 or float16 only");
  }
  if (x.ndim() < 2 || x.shape(0) != 1 || x.shape(1) < 1 ||
      (even && x.shape(1) % 2 != 0)) {
    throw std::invalid_argument(
        std::string(name) + " needs [1, T, ...]" +
        (even ? " with T even" : ""));
  }
}
SealCodec require_codec(const char* name) {
  auto codec = seal_codec();
  if (!codec.has_value()) {
    throw std::runtime_error(
        std::string(name) +
        " no seal codec is installed: a partial never crosses the link in the clear");
  }
  return *codec;
}
} // namespace

array reduce_scatter_sealed(
    const array& x,
    std::optional<Group> group_,
    StreamOrDevice s) {
  auto group = group_.has_value() ? group_.value() : distributed::init();
  require_rows(x, group, "[reduce_scatter_sealed]", true);
  auto codec = require_codec("[reduce_scatter_sealed]");
  auto shape = x.shape();
  shape[1] /= 2;
  auto stream = detail::communication_stream(group, s);
  return array(
      std::move(shape),
      x.dtype(),
      std::make_shared<SealedReduceScatter>(stream, group, codec),
      {x});
}

array all_gather_sealed(
    const array& x,
    std::optional<Group> group_,
    StreamOrDevice s) {
  auto group = group_.has_value() ? group_.value() : distributed::init();
  require_rows(x, group, "[all_gather_sealed]", false);
  auto codec = require_codec("[all_gather_sealed]");
  auto shape = x.shape();
  shape[1] *= 2;
  auto stream = detail::communication_stream(group, s);
  return array(
      std::move(shape),
      x.dtype(),
      std::make_shared<SealedAllGather>(stream, group, codec),
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
#ifdef __APPLE__
  dispatch_apply(spans, DISPATCH_APPLY_AUTO, ^(size_t i) {
    size_t begin = i * kSpan;
    size_t count = std::min(kSpan, n - begin);
    add_span(rank0 + begin, rank1 + begin, out + begin, count);
  });
#else
  for (size_t i = 0; i < spans; ++i) {
    size_t begin = i * kSpan;
    size_t count = std::min(kSpan, n - begin);
    add_span(rank0 + begin, rank1 + begin, out + begin, count);
  }
#endif
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
  if (codec_.streamed() && group().raw_group()->supports_exchange()) {
    eval_streamed(in, out);
    if (!inputs[0].flags().row_contiguous) {
      cpu::get_command_encoder(stream()).add_temporary(in);
    }
    return;
  }
  size_t n = in.nbytes();
  size_t frame = n + codec_.overhead;
  int sealed_len = detail::frame_extent(frame, "[all_sum_sealed]");
  int gathered_len = detail::frame_extent(2 * frame, "[all_sum_sealed]");
  array sealed({sealed_len}, uint8, nullptr, {});
  sealed.set_data(allocator::malloc(frame));
  array gathered({gathered_len}, uint8, nullptr, {});
  gathered.set_data(allocator::malloc(2 * frame));
  array peer(in.shape(), in.dtype(), nullptr, {});
  peer.set_data(allocator::malloc(n));

  int rank = group().rank();
  uint64_t counter = next_sealed_counter();
  auto codec = codec_;
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(sealed);
  encoder.dispatch([codec,
                    src = in.data<char>(),
                    dst = sealed.data<char>(),
                    n,
                    frame,
                    counter]() {
    if (codec.seal(codec.ctx, src, n, dst, counter) != 0) {
      // The gather after this task still runs: never send stale bytes.
      std::memset(dst, 0, frame);
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

// One CPU-stream task that seals, sends, opens and adds piece by piece
// (sealed_stream.h). Each rank receives only the peer's frame.
void SealedAllSum::eval_streamed(const array& in, array& out) {
  detail::PieceLayout layout(in.nbytes(), codec_.piece_bytes);
  if (layout.count > 0xFFFF) {
    throw std::invalid_argument(
        "[all_sum_sealed] a partial needs more than 65,535 sealed pieces");
  }
  detail::AddPieceFn add;
  switch (in.dtype()) {
    case float32:
      add = detail::add_piece<float>;
      break;
    case bfloat16:
      add = detail::add_piece<bfloat16_t>;
      break;
    case float16:
      add = detail::add_piece<float16_t>;
      break;
    default:
      throw std::runtime_error("[all_sum_sealed] unsupported dtype");
  }
  int frame_len = detail::frame_extent(layout.frame(), "[all_sum_sealed]");
  array sealed({frame_len}, uint8, nullptr, {});
  sealed.set_data(allocator::malloc(layout.frame()));
  array received({frame_len}, uint8, nullptr, {});
  received.set_data(allocator::malloc(layout.frame()));

  int rank = group().rank();
  uint64_t counter = next_sealed_counter();
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([codec = codec_,
                    layout,
                    counter,
                    rank,
                    add,
                    group = group().raw_group(),
                    src = in.data<char>(),
                    dst = out.data<char>(),
                    mine = sealed.data<char>(),
                    theirs = received.data<char>()]() {
    detail::StreamedSum sum(
        codec, layout, counter, rank, src, src, dst, mine, theirs, add);
    bool ok = sum.run([&group](
                          const char* send,
                          char* recv,
                          size_t n,
                          const detail::ReadyFn& ready,
                          const detail::ArrivedFn& arrived) {
      group->exchange(send, recv, n, ready, arrived);
    });
    if (!ok) {
      throw std::runtime_error(
          "[all_sum_sealed] a piece did not seal or a peer piece did not "
          "authenticate: refused");
    }
  });
  encoder.add_temporary(sealed);
  encoder.add_temporary(received);
}

void SealedAllSum::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[SealedAllSum::eval_gpu] has no GPU implementation.");
}

namespace {

detail::AddPieceFn add_piece_for(Dtype dtype, const char* name) {
  switch (dtype) {
    case float32:
      return detail::add_piece<float>;
    case bfloat16:
      return detail::add_piece<bfloat16_t>;
    case float16:
      return detail::add_piece<float16_t>;
    default:
      throw std::runtime_error(std::string(name) + " unsupported dtype");
  }
}

// The streamed half-size exchange both new collectives run: `send` (n bytes)
// sealed and sent, the peer's n bytes opened into `out` and, with `add`,
// `addend` added in rank order.
void run_streamed(
    const SealCodec& codec,
    Group& group,
    Stream stream,
    const array& in,
    array& out,
    size_t n,
    const char* send,
    const char* addend,
    char* dst,
    detail::AddPieceFn add,
    const char* name) {
  detail::PieceLayout layout(n, codec.piece_bytes);
  if (layout.count > 0xFFFF) {
    throw std::invalid_argument(
        std::string(name) + " a frame needs more than 65,535 sealed pieces");
  }
  int frame_len = detail::frame_extent(layout.frame(), name);
  array sealed({frame_len}, uint8, nullptr, {});
  sealed.set_data(allocator::malloc(layout.frame()));
  array received({frame_len}, uint8, nullptr, {});
  received.set_data(allocator::malloc(layout.frame()));
  int rank = group.rank();
  uint64_t counter = next_sealed_counter();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  std::string label = name;
  encoder.dispatch([codec,
                    layout,
                    counter,
                    rank,
                    add,
                    label,
                    raw = group.raw_group(),
                    send,
                    addend,
                    dst,
                    mine = sealed.data<char>(),
                    theirs = received.data<char>()]() {
    detail::StreamedSum exchange(
        codec, layout, counter, rank, send, addend, dst, mine, theirs, add);
    bool ok = exchange.run([&raw](
                               const char* sent,
                               char* recv,
                               size_t bytes,
                               const detail::ReadyFn& ready,
                               const detail::ArrivedFn& arrived) {
      raw->exchange(sent, recv, bytes, ready, arrived);
    });
    if (!ok) {
      throw std::runtime_error(
          label +
          " a piece did not seal or a peer piece did not authenticate: refused");
    }
  });
  encoder.add_temporary(sealed);
  encoder.add_temporary(received);
}

// The same without an exchange: seal `send`, gather both frames, open the
// peer's into `dst` and, with `add`, add `addend` in rank order.
void run_gathered(
    const SealCodec& codec,
    Group& group,
    Stream stream,
    const array& in,
    array& out,
    size_t n,
    const char* send,
    const char* addend,
    char* dst,
    detail::AddPieceFn add,
    const char* name) {
  size_t frame = n + codec.overhead;
  int sealed_len = detail::frame_extent(frame, name);
  int gathered_len = detail::frame_extent(2 * frame, name);
  array sealed({sealed_len}, uint8, nullptr, {});
  sealed.set_data(allocator::malloc(frame));
  array gathered({gathered_len}, uint8, nullptr, {});
  gathered.set_data(allocator::malloc(2 * frame));
  int rank = group.rank();
  uint64_t counter = next_sealed_counter();
  std::string label = name;
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(in);
  encoder.set_output_array(sealed);
  encoder.dispatch(
      [codec, send, dst = sealed.data<char>(), n, frame, counter, label]() {
        if (codec.seal(codec.ctx, send, n, dst, counter) != 0) {
          std::memset(dst, 0, frame);
          throw std::runtime_error(label + " the codec refused to seal");
        }
      });
  distributed::detail::all_gather(group, sealed, gathered, stream);
  encoder.set_input_array(gathered);
  encoder.set_output_array(out);
  encoder.dispatch([codec,
                    frames = gathered.data<char>(),
                    addend,
                    dst,
                    n,
                    frame,
                    counter,
                    rank,
                    add,
                    label]() {
    const char* peer_frame = frames + (rank == 0 ? frame : 0);
    if (codec.open(codec.ctx, peer_frame, n, dst, counter, 1 - rank) != 0) {
      throw std::runtime_error(
          label + " a peer frame did not authenticate: refused");
    }
    if (add != nullptr) {
      add(rank, addend, dst, n);
    }
  });
  encoder.add_temporary(sealed);
  encoder.add_temporary(gathered);
}

} // namespace

// Rank r keeps rows [r * T / 2, (r + 1) * T / 2): it sends the peer's half
// of its partial and adds the peer's partial of its own half to its own, in
// rank order (rank 0 + rank 1), as the all-sum does for every element.
void SealedReduceScatter::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto in = inputs[0].flags().row_contiguous
      ? inputs[0]
      : contiguous_copy_cpu(inputs[0], stream());
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  size_t half = out.nbytes();
  int rank = group().rank();
  const char* src = in.data<char>();
  const char* send = src + (rank == 0 ? half : 0);
  const char* mine = src + (rank == 0 ? 0 : half);
  auto add = add_piece_for(in.dtype(), "[reduce_scatter_sealed]");
  auto group = this->group();
  if (codec_.streamed() && group.raw_group()->supports_exchange()) {
    run_streamed(
        codec_,
        group,
        stream(),
        in,
        out,
        half,
        send,
        mine,
        out.data<char>(),
        add,
        "[reduce_scatter_sealed]");
  } else {
    run_gathered(
        codec_,
        group,
        stream(),
        in,
        out,
        half,
        send,
        mine,
        out.data<char>(),
        add,
        "[reduce_scatter_sealed]");
  }
  if (!inputs[0].flags().row_contiguous) {
    cpu::get_command_encoder(stream()).add_temporary(in);
  }
}

void SealedReduceScatter::eval_gpu(
    const std::vector<array>&,
    std::vector<array>&) {
  throw std::runtime_error(
      "[SealedReduceScatter::eval_gpu] has no GPU implementation.");
}

// Rank 0's rows then rank 1's: this rank's rows are copied in place and the
// peer's arrive sealed.
void SealedAllGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto in = inputs[0].flags().row_contiguous
      ? inputs[0]
      : contiguous_copy_cpu(inputs[0], stream());
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  size_t n = in.nbytes();
  int rank = group().rank();
  char* own = out.data<char>() + (rank == 0 ? 0 : n);
  char* theirs = out.data<char>() + (rank == 0 ? n : 0);
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch(
      [src = in.data<char>(), own, n]() { std::memcpy(own, src, n); });
  auto group = this->group();
  if (codec_.streamed() && group.raw_group()->supports_exchange()) {
    run_streamed(
        codec_,
        group,
        stream(),
        in,
        out,
        n,
        in.data<char>(),
        nullptr,
        theirs,
        nullptr,
        "[all_gather_sealed]");
  } else {
    run_gathered(
        codec_,
        group,
        stream(),
        in,
        out,
        n,
        in.data<char>(),
        nullptr,
        theirs,
        nullptr,
        "[all_gather_sealed]");
  }
  if (!inputs[0].flags().row_contiguous) {
    encoder.add_temporary(in);
  }
}

void SealedAllGather::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[SealedAllGather::eval_gpu] has no GPU implementation.");
}

} // namespace mlx::core::distributed
