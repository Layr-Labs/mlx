// Copyright © 2026 Darkbloom.

// Tests of the two-rank mesh exchange and of the streamed sealed sum with
// simulated verbs. The codec here is a stand-in: a keyed XOR stream and a
// 16-byte keyed hash per piece. It finds changed bytes, wrong pieces and wrong
// counters, as the real AES-GCM codec does; it is not cryptography.

#include <iostream>
#include <random>
#include <thread>

#include "fixture.h"
#include "mlx/distributed/sealed_stream.h"

namespace {

using mlx::core::bfloat16_t;
using mlx::core::float16_t;
namespace sealed = mlx::core::distributed;
namespace detail = mlx::core::distributed::detail;

constexpr int64_t kFrame = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));

int64_t class_bytes(int64_t length) {
  int64_t bytes = FRAME_SIZE;
  while (bytes < length && bytes < kFrame) {
    bytes *= 2;
  }
  return bytes;
}

std::vector<char> pattern(int64_t n, int seed) {
  std::vector<char> v(n);
  std::mt19937 random(seed);
  for (auto& c : v) {
    c = char(random());
  }
  return v;
}

// Frames: full ones, then the last in the smallest buffer that holds it.
void exchange_frames(const std::vector<fixture::Frame>& frames, int64_t n) {
  const int64_t count = (n + kFrame - 1) / kFrame;
  fixture::require(int64_t(frames.size()) == count, "Unexpected frame count");
  for (int64_t j = 0; j < count; ++j) {
    const int64_t length = std::min(kFrame, n - j * kFrame);
    fixture::require(
        int64_t(frames[j].bytes.size()) == class_bytes(length),
        "Unexpected frame size class");
    fixture::zero_tail(frames[j], length);
  }
}

void exchange_sizes() {
  for (int64_t n :
       {int64_t(1),
        int64_t(16),
        int64_t(4096),
        int64_t(4097),
        int64_t(8208),
        kFrame,
        kFrame + 1,
        2 * kFrame,
        3 * kFrame + 2064}) {
    fixture::MeshPair pair;
    std::array<std::vector<char>, 2> input{pattern(n, 1), pattern(n, 2)};
    std::array<std::vector<char>, 2> output{
        std::vector<char>(n + 1, 0x29), std::vector<char>(n + 1, 0x29)};
    std::array<std::vector<size_t>, 2> arrived;
    auto run = [&](int rank) {
      // Each frame is refused twice before it is ready.
      int polls = 0;
      pair.nodes[rank]->exchange(
          input[rank].data(),
          output[rank].data(),
          n,
          [&](size_t) { return ++polls % 3 == 0; },
          [&, rank](size_t end) { arrived[rank].push_back(end); });
    };
    fixture::both([&] { run(0); }, [&] { run(1); });
    for (int rank = 0; rank < 2; ++rank) {
      const auto& peer = input[1 - rank];
      fixture::require(
          std::equal(peer.begin(), peer.end(), output[rank].begin()) &&
              output[rank].back() == 0x29,
          "Exchange changed the peer's bytes or wrote past the end");
      const auto& ends = arrived[rank];
      fixture::require(
          !ends.empty() && ends.back() == size_t(n) &&
              std::is_sorted(ends.begin(), ends.end()) &&
              std::adjacent_find(ends.begin(), ends.end()) == ends.end(),
          "arrived() must report increasing ends up to the message size");
      exchange_frames(fixture::selected(pair.fabric, rank, rank, 0), n);
    }
    pair.fabric.require_drained();
  }
}

// A frame leaves only when ready() says its bytes are final.
void exchange_waits_for_ready() {
  const int64_t n = 5 * kFrame + 100;
  fixture::MeshPair pair;
  pair.fabric.idle_wait = std::chrono::milliseconds(0);
  std::array<std::vector<char>, 2> final{pattern(n, 3), pattern(n, 4)};
  std::array<std::vector<char>, 2> input{
      std::vector<char>(n, char(fixture::sentinel)),
      std::vector<char>(n, char(fixture::sentinel))};
  std::array<std::vector<char>, 2> output{
      std::vector<char>(n), std::vector<char>(n)};
  std::array<std::atomic<size_t>, 2> written{0, 0};
  auto run = [&](int rank) {
    std::thread writer([&, rank] {
      for (int64_t at = 0; at < n; at += 300000) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        int64_t end = std::min(n, at + 300000);
        std::copy(
            final[rank].begin() + at,
            final[rank].begin() + end,
            input[rank].begin() + at);
        written[rank].store(end, std::memory_order_release);
      }
    });
    try {
      pair.nodes[rank]->exchange(
          input[rank].data(),
          output[rank].data(),
          n,
          [&, rank](size_t end) {
            return written[rank].load(std::memory_order_acquire) >= end;
          },
          [](size_t) {});
    } catch (...) {
      writer.join();
      throw;
    }
    writer.join();
  };
  fixture::both([&] { run(0); }, [&] { run(1); });
  fixture::require(
      output[0] == final[1] && output[1] == final[0],
      "A frame left before its bytes were final");
  pair.fabric.require_drained();
}

// A failed completion closes the group: the call throws, later calls refuse.
void exchange_failure() {
  const int64_t n = 3 * kFrame;
  fixture::MeshPair pair;
  pair.fabric.expire_after(std::chrono::milliseconds(1000));
  fixture::endpoint(pair.links[1][0]).fault.fail_receive = 1;
  std::array<std::vector<char>, 2> input{pattern(n, 5), pattern(n, 6)};
  std::array<std::vector<char>, 2> output{
      std::vector<char>(n), std::vector<char>(n)};
  std::array<std::string, 2> errors;
  auto run = [&](int rank) {
    try {
      pair.nodes[rank]->exchange(
          input[rank].data(),
          output[rank].data(),
          n,
          [](size_t) { return true; },
          [](size_t) {});
    } catch (const std::exception& error) {
      errors[rank] = error.what();
    }
  };
  fixture::both([&] { run(0); }, [&] { run(1); });
  // Connection::poll (upstream #4767) raises the failed completion.
  fixture::require(
      errors[1].rfind("[jaccl] Work completion failed with status", 0) == 0,
      "A failed completion must fail the exchange");
  std::string later;
  try {
    pair.nodes[1]->exchange(
        input[1].data(),
        output[1].data(),
        16,
        [](size_t) { return true; },
        [](size_t) {});
  } catch (const std::exception& error) {
    later = error.what();
  }
  fixture::require(
      later.find("closed") != std::string::npos,
      "A closed group must refuse a later exchange");
}

// The stand-in codec.
struct Codec {
  int rank;
  std::atomic<bool> refused{false};
  int64_t refuse_seal = -1;
  int64_t tamper = -1;
  bool jitter = false;

  static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }
  static uint64_t key(uint64_t counter, int sender, uint32_t piece) {
    return mix(counter * 0x10001 ^ uint64_t(sender) << 60 ^ uint64_t(piece));
  }
  static void crypt(const char* in, char* out, size_t len, uint64_t key) {
    for (size_t i = 0; i < len; ++i) {
      out[i] = char(in[i] ^ char(mix(key ^ (i / 8)) >> (8 * (i % 8))));
    }
  }
  static void tag(
      const char* ct,
      size_t len,
      uint64_t key,
      uint32_t pieces,
      size_t total,
      char* out) {
    uint64_t a = mix(key ^ pieces), b = mix(key ^ total ^ 0x5555);
    for (size_t i = 0; i < len; ++i) {
      a = mix(a ^ uint8_t(ct[i]));
      b = b * 0x100000001b3ull ^ uint8_t(ct[i]);
    }
    std::memcpy(out, &a, 8);
    std::memcpy(out + 8, &b, 8);
  }

  static int seal(
      void* ctx,
      const void* plaintext,
      size_t len,
      void* sealed,
      uint64_t counter,
      uint32_t piece,
      uint32_t pieces,
      size_t total) {
    auto& self = *static_cast<Codec*>(ctx);
    if (self.refused || int64_t(piece) == self.refuse_seal) {
      return 1;
    }
    if (self.jitter) {
      std::this_thread::sleep_for(std::chrono::microseconds(piece % 5 * 40));
    }
    uint64_t k = key(counter, self.rank, piece);
    auto out = static_cast<char*>(sealed);
    crypt(static_cast<const char*>(plaintext), out, len, k);
    tag(out, len, k, pieces, total, out + len);
    if (int64_t(piece) == self.tamper) {
      out[len / 2] ^= 1;
    }
    return 0;
  }

  static int open(
      void* ctx,
      const void* sealed,
      size_t len,
      void* plaintext,
      uint64_t counter,
      uint32_t piece,
      uint32_t pieces,
      size_t total,
      int peer) {
    auto& self = *static_cast<Codec*>(ctx);
    if (self.refused || peer != 1 - self.rank) {
      return 1;
    }
    uint64_t k = key(counter, peer, piece);
    auto in = static_cast<const char*>(sealed);
    char expected[sealed::kSealTagBytes];
    tag(in, len, k, pieces, total, expected);
    if (std::memcmp(expected, in + len, sizeof(expected)) != 0) {
      self.refused = true;
      return 1;
    }
    crypt(in, static_cast<char*>(plaintext), len, k);
    return 0;
  }

  sealed::SealCodec codec(size_t piece_bytes) {
    sealed::SealCodec c;
    c.seal_piece = seal;
    c.open_piece = open;
    c.ctx = this;
    c.piece_bytes = piece_bytes;
    return c;
  }
};

struct SumResult {
  std::array<bool, 2> ok{};
  std::array<std::vector<char>, 2> out;
  std::array<std::vector<char>, 2> partial;
};

template <typename T>
SumResult streamed_sum(
    fixture::MeshPair& pair,
    std::array<Codec, 2>& codecs,
    size_t elements,
    size_t piece_bytes,
    uint64_t counter) {
  SumResult r;
  const size_t n = elements * sizeof(T);
  detail::PieceLayout layout(n, piece_bytes);
  auto run = [&](int rank) {
    r.partial[rank] = pattern(n, 10 + rank);
    r.out[rank].assign(n, char(fixture::sentinel));
    // Stale bytes, as a reused allocation can hold.
    std::vector<char> mine(layout.frame(), char(fixture::sentinel));
    std::vector<char> theirs(layout.frame(), char(fixture::sentinel));
    detail::StreamedSum sum(
        codecs[rank].codec(piece_bytes),
        layout,
        counter,
        rank,
        r.partial[rank].data(),
        r.partial[rank].data(),
        r.out[rank].data(),
        mine.data(),
        theirs.data(),
        detail::add_piece<T>);
    r.ok[rank] = sum.run([&, rank](
                             const char* send,
                             char* recv,
                             size_t bytes,
                             const detail::ReadyFn& ready,
                             const detail::ArrivedFn& arrived) {
      pair.nodes[rank]->exchange(send, recv, bytes, ready, arrived);
    });
  };
  fixture::both([&] { run(0); }, [&] { run(1); });
  pair.fabric.require_drained();
  return r;
}

// The gathered path's sum: rank 0's addend + rank 1's, widened, rounded once.
template <typename T>
std::vector<char> reference(const SumResult& r) {
  std::vector<char> sum(r.partial[0].size());
  auto a = reinterpret_cast<const T*>(r.partial[0].data());
  auto b = reinterpret_cast<const T*>(r.partial[1].data());
  auto out = reinterpret_cast<T*>(sum.data());
  for (size_t i = 0; i < sum.size() / sizeof(T); ++i) {
    out[i] =
        static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
  }
  return sum;
}

template <typename T>
void identical(size_t elements, size_t piece_bytes, bool jitter) {
  fixture::MeshPair pair;
  pair.fabric.idle_wait = std::chrono::milliseconds(0);
  std::array<Codec, 2> codecs;
  codecs[0].rank = 0;
  codecs[1].rank = 1;
  codecs[0].jitter = codecs[1].jitter = jitter;
  auto r = streamed_sum<T>(pair, codecs, elements, piece_bytes, 7);
  auto expected = reference<T>(r);
  fixture::require(r.ok[0] && r.ok[1], "A streamed sum was refused");
  fixture::require(
      r.out[0] == expected && r.out[1] == expected,
      "The streamed sum differs from the gathered sum, bit for bit");
}

void streamed_sum_identical() {
  constexpr size_t piece = (256 << 10) - sealed::kSealTagBytes;
  identical<bfloat16_t>(4096, piece, false); // decode: one piece
  identical<float>(4096, piece, false);
  identical<bfloat16_t>(2048 * 1024, piece, false); // 4 MiB, 17 pieces
  identical<float16_t>(2048 * 1024, piece, true);
  identical<float>(1 << 20, 100000, true); // pieces that straddle frames
  identical<bfloat16_t>(4096 * 2048, piece, false); // 16 MiB, 65 pieces
}

void streamed_sum_tamper() {
  fixture::MeshPair pair;
  std::array<Codec, 2> codecs;
  codecs[0].rank = 0;
  codecs[1].rank = 1;
  codecs[0].tamper = 5;
  auto r = streamed_sum<float>(
      pair, codecs, 1 << 20, (256 << 10) - sealed::kSealTagBytes, 9);
  fixture::require(
      !r.ok[1] && codecs[1].refused,
      "A changed byte in a peer piece must refuse the sum");
}

void streamed_sum_seal_failure() {
  constexpr size_t piece = (256 << 10) - sealed::kSealTagBytes;
  fixture::MeshPair pair;
  std::array<Codec, 2> codecs;
  codecs[0].rank = 0;
  codecs[1].rank = 1;
  codecs[0].refuse_seal = 3;
  auto r = streamed_sum<float>(pair, codecs, 1 << 20, piece, 11);
  fixture::require(
      !r.ok[0] && !r.ok[1], "A piece that did not seal must fail both ranks");
  // The piece that did not seal went out as zeros, not as stale bytes.
  std::vector<char> wire;
  for (const auto& f : fixture::selected(pair.fabric, 0, 0, 0)) {
    wire.insert(wire.end(), f.bytes.begin(), f.bytes.end());
  }
  detail::PieceLayout layout(size_t(1 << 20) * sizeof(float), piece);
  fixture::require(
      std::all_of(
          wire.begin() + layout.sealed(3),
          wire.begin() + layout.sealed_end(3),
          [](char c) { return c == 0; }),
      "A piece that did not seal must cross as zeros");
}

void piece_layout() {
  constexpr size_t piece = (256 << 10) - sealed::kSealTagBytes;
  detail::PieceLayout bf16(size_t(16) << 20, piece);
  fixture::require(
      bf16.count == 65 && bf16.sealed(4) == size_t(kFrame) &&
          bf16.frame() == (size_t(16) << 20) + 65 * 16,
      "Four sealed pieces fill each 1 MiB frame");
  detail::PieceLayout decode(8192, piece);
  fixture::require(
      decode.count == 1 && decode.frame() == 8208, "A decode sum is one piece");
  detail::PieceLayout empty(0, piece);
  fixture::require(
      empty.count == 1 && empty.length(0) == 0 && empty.frame() == 16,
      "An empty partial is one empty piece");
}

bool group(void (*check)(), const char* name) {
  try {
    check();
    std::cout << "PASS " << name << '\n';
    return true;
  } catch (const std::exception& error) {
    std::cout << "FAIL " << name << ": " << error.what() << '\n';
    return false;
  }
}

} // namespace

int main() {
  bool passed = group(piece_layout, "streamed piece layout");
  passed &= group(exchange_sizes, "exchange sizes, tail class, arrived ends");
  passed &= group(exchange_waits_for_ready, "exchange sends only ready bytes");
  passed &= group(
      streamed_sum_identical, "streamed sum bit-identical to gathered sum");
  passed &= group(streamed_sum_tamper, "streamed sum refuses a changed byte");
  passed &=
      group(streamed_sum_seal_failure, "streamed sum seal failure sends zeros");
  passed &= group(exchange_failure, "exchange failure closes the group");
  const auto& ledger = fixture::ledger();
  if (ledger.queue_pairs != 0 || ledger.completion_queues != 0 ||
      ledger.regions != 0) {
    std::cout << "FAIL fake verbs objects remain after the checks\n";
    passed = false;
  }
  return passed ? 0 : 1;
}
