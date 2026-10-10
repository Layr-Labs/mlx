// Copyright © 2026 Apple Inc.

// Tests of the send staging with simulated verbs. The scratch buffers start
// with a sentinel in each byte, and the fake endpoint keeps each posted frame.
// Each frame must hold the payload and then only zero bytes.

#include <iostream>
#include <limits>

#include "fixture.h"
#include "jaccl/send_frame.h"

namespace {

void staging_bounds() {
  alignas(double) std::array<char, 32> frame;
  const std::array<uint32_t, 2> input{0x01020304, 0x05060708};
  frame.fill(char(fixture::sentinel));
  jaccl::stage_send_frame(std::span<char>(frame), input.data(), 2);
  fixture::require(
      std::memcmp(frame.data(), input.data(), 8) == 0,
      "Typed payload bytes changed");
  fixture::zero_tail(
      {0, 0, 0, std::vector<char>(frame.begin(), frame.end())}, 8);
  jaccl::stage_send_frame(
      std::span<char>(frame), static_cast<const uint32_t*>(nullptr), 0);
  fixture::zero_tail(
      {0, 0, 0, std::vector<char>(frame.begin(), frame.end())}, 0);
  for (int64_t count :
       {int64_t(-1), int64_t(9), std::numeric_limits<int64_t>::max()}) {
    frame.fill(char(fixture::sentinel));
    bool refused = false;
    try {
      jaccl::stage_send_frame(std::span<char>(frame), input.data(), count);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    fixture::require(
        refused &&
            std::all_of(
                frame.begin(),
                frame.end(),
                [](char value) {
                  return static_cast<unsigned char>(value) == fixture::sentinel;
                }),
        "Invalid count must refuse before any scratch write");
  }
  bool refused = false;
  try {
    jaccl::stage_send_frame(
        std::span<char>(frame), static_cast<const char*>(nullptr), 1);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  fixture::require(refused, "Null nonempty source refused");
}

void mesh_point_to_point() {
  fixture::MeshPair pair;
  const int64_t maximum = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));
  for (int64_t count :
       {int64_t(1),
        int64_t(24),
        int64_t(4096),
        int64_t(17),
        int64_t(4097),
        maximum * 3 + 17}) {
    pair.fabric.frames.clear();
    std::vector<char> input(count, 0x53), output(count + 1, 0x29);
    fixture::both(
        [&] { pair.nodes[0]->send(input.data(), count, 1); },
        [&] { pair.nodes[1]->recv(output.data(), count, 0); });
    fixture::require(
        std::equal(input.begin(), input.end(), output.begin()) &&
            output.back() == 0x29,
        "Mesh receive changed payload or wrote past end");
    pair.fabric.require_drained();
    fixture::payload_frames(fixture::selected(pair.fabric, 0, 0, 0), input);
  }
}

void ring_point_to_point() {
  const int64_t maximum = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));
  for (int wires : {1, 2}) {
    fixture::RingPair pair(wires);
    for (int64_t count :
         {int64_t(1), int64_t(4096), int64_t(17), maximum * 3 * wires + 13}) {
      pair.fabric.frames.clear();
      std::vector<char> input(count, 0x64), output(count + 1, 0x29);
      fixture::both(
          [&] { pair.nodes[0]->send(input.data(), count, 1, wires); },
          [&] { pair.nodes[1]->recv(output.data(), count, 0, wires); });
      fixture::require(
          std::equal(input.begin(), input.end(), output.begin()) &&
              output.back() == 0x29,
          "Ring receive changed payload or wrote past end");
      pair.fabric.require_drained();
      const size_t per_wire = (count + wires - 1) / wires;
      for (int wire = 0; wire < wires; ++wire) {
        const size_t start = std::min(size_t(count), wire * per_wire);
        const size_t end = std::min(size_t(count), (wire + 1) * per_wire);
        auto frames = fixture::selected(pair.fabric, 0, 1, wire);
        fixture::payload_frames(
            frames, std::span<const char>(input).subspan(start, end - start));
      }
    }
  }
}

std::span<const char> byte_view(
    const std::vector<float>& values,
    size_t count) {
  return {reinterpret_cast<const char*>(values.data()), count * sizeof(float)};
}

void all_equal(const std::vector<float>& values, size_t count, float expected) {
  fixture::require(
      std::all_of(
          values.begin(),
          values.begin() + count,
          [&](float value) { return value == expected; }),
      "Collective logical output changed");
  fixture::require(
      values[count] == -19, "Collective wrote outside logical output");
}

void mesh_collectives() {
  const size_t count =
      (FRAME_SIZE * (1 << (BUFFER_SIZES - 1)) * 3) / sizeof(float) + 7;
  for (int operation : {0, 1, 2}) {
    fixture::MeshPair pair;
    std::array<std::vector<float>, 2> input, output;
    for (int rank = 0; rank < 2; ++rank) {
      input[rank].assign(count * 2, float(rank + 1));
      output[rank].assign(count * 2 + 1, -19);
    }
    auto run = [&](int rank) {
      if (operation == 0) {
        pair.nodes[rank]->all_reduce(
            input[rank].data(), output[rank].data(), count, fixture::Add{});
      } else if (operation == 1) {
        pair.nodes[rank]->all_gather(
            reinterpret_cast<const char*>(input[rank].data()),
            reinterpret_cast<char*>(output[rank].data()),
            count * sizeof(float));
      } else {
        pair.nodes[rank]->sum_scatter(
            input[rank].data(), output[rank].data(), count, fixture::Add{});
      }
    };
    fixture::both([&] { run(0); }, [&] { run(1); });
    pair.fabric.require_drained();
    for (int rank = 0; rank < 2; ++rank) {
      if (operation == 1) {
        fixture::require(
            std::all_of(
                output[rank].begin(),
                output[rank].begin() + count,
                [](float value) { return value == 1; }),
            "Gather rank0 payload changed");
        fixture::require(
            std::all_of(
                output[rank].begin() + count,
                output[rank].begin() + count * 2,
                [](float value) { return value == 2; }) &&
                output[rank].back() == -19,
            "Gather rank1 payload or output bound changed");
      } else {
        all_equal(output[rank], count, 3);
      }
      fixture::payload_frames(
          fixture::selected(pair.fabric, rank, rank, 0),
          byte_view(input[rank], count));
    }
  }
}

void ring_collectives() {
  // A chunk of 1 gives slices with no payload. The long chunk fills each
  // pipeline slot again. buffer_size_from_message selects the frame sizes.
  const int64_t maximum = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));
  for (bool scatter : {false, true}) {
    for (int64_t chunk :
         {int64_t(1), maximum * 3 / int64_t(sizeof(float)) * 4 + 7}) {
      const int wires = 2;
      fixture::RingPair pair(wires);
      const int64_t count = scatter ? chunk * 2 : chunk;
      const int64_t stage_chunk = scatter ? chunk : (count + 1) / 2;
      const int64_t per_wire = (stage_chunk + 2 * wires - 1) / (2 * wires);
      const int64_t capacity =
          buffer_size_from_message(per_wire * sizeof(float)).second;
      const int64_t elements = capacity / sizeof(float);
      const int64_t steps = (per_wire + elements - 1) / elements;
      std::array<std::vector<float>, 2> input, output;
      for (int rank = 0; rank < 2; ++rank) {
        input[rank].assign(count, float(rank + 1));
        output[rank].assign((scatter ? chunk : count) + 32, -19);
      }
      auto run = [&](int rank) {
        if (scatter) {
          pair.nodes[rank]->reduce_scatter(
              input[rank].data(),
              output[rank].data(),
              count,
              wires,
              fixture::Add{});
        } else {
          pair.nodes[rank]->all_reduce<2>(
              input[rank].data(),
              output[rank].data(),
              count,
              wires,
              fixture::Add{});
        }
      };
      fixture::both([&] { run(0); }, [&] { run(1); });
      pair.fabric.require_drained();
      for (int rank = 0; rank < 2; ++rank) {
        all_equal(output[rank], scatter ? chunk : count, 3);
        for (int direction = 0; direction < 2; ++direction) {
          for (int wire = 0; wire < wires; ++wire) {
            auto frames = fixture::selected(pair.fabric, rank, direction, wire);
            const int passes = scatter ? 1 : 2;
            fixture::require(
                frames.size() == size_t(steps * passes),
                "Ring completion/frame count changed");
            const int64_t offset =
                direction * wires * per_wire + wire * per_wire;
            const int64_t end = std::min(stage_chunk, offset + per_wire);
            for (int pass = 0; pass < passes; ++pass) {
              const int64_t source_offset =
                  (pass == 0 ? rank : 1 - rank) * stage_chunk;
              const int64_t limit = scatter
                  ? end
                  : std::min(end, std::max<int64_t>(0, count - source_offset));
              for (int64_t step = 0; step < steps; ++step) {
                const int64_t valid = std::max<int64_t>(
                    0, std::min(elements, limit - offset - step * elements));
                const auto& frame = frames[pass * steps + step];
                fixture::require(
                    frame.bytes.size() == size_t(capacity),
                    "Ring posted SGE length changed");
                fixture::zero_tail(frame, valid * sizeof(float));
              }
            }
          }
        }
      }
    }
  }
}

// Ring all gather with different data on each rank. A gather of equal data
// cannot find a region that was not transferred.
void ring_gather() {
  const int64_t maximum = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));
  for (int wires : {1, 2}) {
    for (int64_t bytes :
         {int64_t(1),
          int64_t(7),
          int64_t(4096),
          int64_t(4097),
          maximum * 3 * wires + 13}) {
      fixture::RingPair pair(wires);
      std::array<std::vector<char>, 2> input, output;
      for (int rank = 0; rank < 2; ++rank) {
        input[rank].resize(bytes);
        for (int64_t index = 0; index < bytes; ++index) {
          input[rank][index] = char((index * 31 + rank * 101 + 7) % 251);
        }
        output[rank].assign(bytes * 2 + 1, 0x29);
      }
      fixture::both(
          [&] {
            pair.nodes[0]->all_gather(
                input[0].data(), output[0].data(), bytes, wires);
          },
          [&] {
            pair.nodes[1]->all_gather(
                input[1].data(), output[1].data(), bytes, wires);
          });
      pair.fabric.require_drained();
      for (int rank = 0; rank < 2; ++rank) {
        for (int source = 0; source < 2; ++source) {
          fixture::require(
              std::equal(
                  input[source].begin(),
                  input[source].end(),
                  output[rank].begin() + source * bytes),
              "Ring all gather lost or misplaced a rank region");
        }
        fixture::require(
            output[rank].back() == 0x29,
            "Ring all gather wrote past the output");
      }
    }
  }
}

// Run all groups, so that a failure does not hide the next ones.
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
  bool passed =
      group(staging_bounds, "bounded typed staging and zero-logical frame");
  passed &= group(
      mesh_point_to_point,
      "mesh short/full/reused/multiframe and actual receive");
  passed &= group(
      ring_point_to_point,
      "ring one/two-wire short/full/reused/multiframe receive");
  passed &= group(
      mesh_collectives, "mesh gather/reduce/scatter valid payload and tail");
  passed &= group(
      ring_collectives, "ring reduction/scatter full/partial/zero slices");
  passed &=
      group(ring_gather, "ring all_gather distinct rank data, one/two wires");
  // The completion queues of the groups have COMPLETION_QUEUE_DEPTH entries,
  // four times the work a queue pair can have outstanding. Report how many
  // completions this traffic queued at the same time; it must stay within the
  // outstanding bound, so that the headroom is real.
  static_assert(COMPLETION_QUEUE_DEPTH == 4 * (MAX_SEND_WR + MAX_RECV_WR));
  const size_t depth = fixture::ledger().completion_high_water;
  const size_t outstanding = MAX_SEND_WR + MAX_RECV_WR;
  if (depth <= outstanding) {
    std::cout << "PASS completion queue high-water mark " << depth << " of "
              << COMPLETION_QUEUE_DEPTH << " entries (outstanding bound "
              << outstanding << ")\n";
  } else {
    std::cout << "FAIL completion queue high-water mark " << depth
              << " exceeds the outstanding bound " << outstanding << '\n';
    passed = false;
  }
  const auto& ledger = fixture::ledger();
  if (ledger.queue_pairs == 0 && ledger.completion_queues == 0 &&
      ledger.regions == 0) {
    std::cout
        << "PASS every fake queue pair, completion queue and region was freed once\n";
  } else {
    std::cout << "FAIL fake verbs objects remain after the checks\n";
    passed = false;
  }
  if (!passed) {
    return 1;
  }
  return 0;
}
