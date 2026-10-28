// Copyright © 2026 Darkbloom.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/distributed/sealed_stream.h"

namespace {
namespace sealed = mlx::core::distributed;
namespace detail = mlx::core::distributed::detail;

// A deterministic test codec, not encryption. Tags bind all callback fields.
struct Codec {
  int rank;
  size_t total;
  int fail_seal = -1;
  std::atomic<int> active{0};
  std::atomic<int> seals{0};
  std::atomic<int> opens{0};
  std::atomic<bool> bad_identity{false};
  std::atomic<bool> changed_thread{false};
  const std::thread::id caller = std::this_thread::get_id();
  static constexpr uint64_t counter = 19;

  bool identity(
      size_t bytes,
      uint64_t count,
      uint32_t piece,
      uint32_t pieces,
      size_t whole) {
    const size_t expected_pieces = std::max(size_t(1), (total + 4095) / 4096);
    bool valid = count == counter && whole == total &&
        pieces == expected_pieces && piece < expected_pieces &&
        size_t(piece) * 4096 <= total;
    valid =
        valid && bytes == std::min(size_t(4096), total - size_t(piece) * 4096);
    if (!valid)
      bad_identity = true;
    if (caller != std::this_thread::get_id())
      changed_thread = true;
    return valid;
  }

  static unsigned char tag(
      uint64_t counter,
      int rank,
      uint32_t piece,
      uint32_t pieces,
      size_t total,
      size_t index) {
    return static_cast<unsigned char>(
        counter + rank + piece + pieces + total + index);
  }

  static int seal(
      void* context,
      const void* input,
      size_t bytes,
      void* output,
      uint64_t count,
      uint32_t piece,
      uint32_t pieces,
      size_t total) {
    auto& self = *static_cast<Codec*>(context);
    ++self.active;
    ++self.seals;
    if (!self.identity(bytes, count, piece, pieces, total) ||
        int(piece) == self.fail_seal) {
      --self.active;
      return 1;
    }
    auto in = static_cast<const unsigned char*>(input);
    auto out = static_cast<unsigned char*>(output);
    for (size_t i = 0; i < bytes; ++i)
      out[i] = in[i] ^ (self.rank + 41);
    for (size_t i = 0; i < sealed::kSealTagBytes; ++i)
      out[bytes + i] = tag(count, self.rank, piece, pieces, total, i);
    --self.active;
    return 0;
  }

  static int open(
      void* context,
      const void* input,
      size_t bytes,
      void* output,
      uint64_t count,
      uint32_t piece,
      uint32_t pieces,
      size_t total,
      int peer) {
    auto& self = *static_cast<Codec*>(context);
    ++self.active;
    ++self.opens;
    auto in = static_cast<const unsigned char*>(input);
    auto out = static_cast<unsigned char*>(output);
    bool valid = self.identity(bytes, count, piece, pieces, total) &&
        peer == 1 - self.rank;
    for (size_t i = 0; i < sealed::kSealTagBytes; ++i)
      valid &= in[bytes + i] == tag(count, peer, piece, pieces, total, i);
    if (valid)
      for (size_t i = 0; i < bytes; ++i)
        out[i] = in[i] ^ (peer + 41);
    --self.active;
    return valid ? 0 : 1;
  }

  sealed::SealCodec value() {
    sealed::SealCodec result;
    result.ctx = this;
    result.seal_piece = seal;
    result.open_piece = open;
    result.piece_bytes = 4096;
    return result;
  }
};

struct Fixture {
  Codec local;
  Codec peer;
  detail::PieceLayout layout;
  std::vector<float> send, addend, remote, output;
  std::vector<char> sent_frame, received_frame, peer_frame;

  Fixture(int rank, size_t count)
      : local{rank, count * sizeof(float)},
        peer{1 - rank, count * sizeof(float)},
        layout(count * sizeof(float), 4096),
        send(std::max(size_t(1), count)),
        addend(std::max(size_t(1), count)),
        remote(std::max(size_t(1), count)),
        output(std::max(size_t(1), count), -100),
        sent_frame(layout.frame(), char(0x5a)),
        received_frame(layout.frame()),
        peer_frame(layout.frame()) {
    for (size_t i = 0; i < count; ++i) {
      send[i] = float(i % 31) / 8;
      addend[i] = float(i % 17) / 4;
      remote[i] = float(i % 13) / 2;
    }
    for (size_t k = 0; k < layout.count; ++k)
      Codec::seal(
          &peer,
          reinterpret_cast<const char*>(remote.data()) + layout.plain(k),
          layout.length(k),
          peer_frame.data() + layout.sealed(k),
          Codec::counter,
          uint32_t(k),
          uint32_t(layout.count),
          layout.bytes);
  }

  bool run(bool add, bool throw_exchange = false) {
    detail::StreamedSum sum(
        local.value(),
        layout,
        Codec::counter,
        local.rank,
        reinterpret_cast<const char*>(send.data()),
        reinterpret_cast<const char*>(addend.data()),
        reinterpret_cast<char*>(output.data()),
        sent_frame.data(),
        received_frame.data(),
        add ? detail::add_piece<float> : nullptr);
    return sum.run([&](const char*,
                       char* received,
                       size_t bytes,
                       const detail::ReadyFn& ready,
                       const detail::ArrivedFn& arrived) {
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!ready(bytes)) {
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("test readiness timeout");
        std::this_thread::yield();
      }
      std::memcpy(received, peer_frame.data(), bytes);
      for (size_t k = 0; k < layout.count; ++k) {
        arrived(layout.sealed_end(k) - 1);
        CHECK(local.opens <= int(k));
        arrived(layout.sealed_end(k));
        if (throw_exchange)
          throw std::runtime_error("test exchange failure");
      }
    });
  }
};
} // namespace

TEST_CASE("sealed pieces preserve exact rank sum and gathered bytes") {
  for (int rank : {0, 1}) {
    for (size_t count : {size_t(0), size_t(17), size_t(4097)}) {
      for (bool add : {false, true}) {
        Fixture fixture(rank, count);
        CHECK(fixture.run(add));
        std::vector<float> expected(count);
        for (size_t i = 0; i < count; ++i)
          expected[i] = !add ? fixture.remote[i]
              : rank == 0    ? fixture.addend[i] + fixture.remote[i]
                             : fixture.remote[i] + fixture.addend[i];
        if (count != 0)
          CHECK(
              std::memcmp(
                  expected.data(),
                  fixture.output.data(),
                  count * sizeof(float)) == 0);
        CHECK(fixture.local.active == 0);
        CHECK(fixture.local.seals == fixture.layout.count);
        CHECK(fixture.local.opens == fixture.layout.count);
        CHECK_FALSE(fixture.local.bad_identity);
#ifndef __APPLE__
        CHECK_FALSE(fixture.local.changed_thread);
#endif
      }
    }
  }
}

TEST_CASE("sealed pieces refuse changed peer tags and zero failed seals") {
  Fixture tampered(0, 4097);
  tampered.peer_frame[tampered.layout.sealed_end(1) - 1] ^= 1;
  CHECK_FALSE(tampered.run(true));
  CHECK(tampered.local.active == 0);
  for (size_t i = tampered.layout.plain(1) / sizeof(float); i <
       (tampered.layout.plain(1) + tampered.layout.length(1)) / sizeof(float);
       ++i)
    CHECK(tampered.output[i] == -100);
  Fixture failed(1, 4097);
  failed.local.fail_seal = 1;
  CHECK_FALSE(failed.run(true));
  CHECK(failed.local.active == 0);
  for (size_t i = failed.layout.sealed(1); i < failed.layout.sealed_end(1); ++i)
    CHECK(failed.sent_frame[i] == 0);
}

TEST_CASE("sealed exchange errors leave no callback owner running") {
  Fixture fixture(0, 4097);
  CHECK_THROWS_WITH(fixture.run(true, true), "test exchange failure");
  CHECK(fixture.local.active == 0);
  CHECK(fixture.local.seals == fixture.layout.count);
  CHECK(fixture.local.opens == 1);
}

TEST_CASE("sealed frames longer than an int array shape are refused") {
  const size_t longest = std::numeric_limits<int>::max();
  CHECK(detail::frame_extent(0, "[test]") == 0);
  CHECK(
      detail::frame_extent(longest, "[test]") ==
      std::numeric_limits<int>::max());
  CHECK_THROWS_WITH_AS(
      detail::frame_extent(longest + 1, "[test]"),
      "[test] a sealed frame is longer than 2^31 - 1 bytes",
      std::invalid_argument);
  // A float32 partial of 1 GiB gathers two frames of more than 1 GiB.
  CHECK_THROWS_AS(
      detail::frame_extent(2 * ((size_t(1) << 30) + 16), "[test]"),
      std::invalid_argument);
}
