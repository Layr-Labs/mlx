// Copyright © 2026 Eigen Labs Inc.
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mlx/memory.h"
#include "mlx/mlx.h"

using namespace mlx::core;

// Standalone so the caller can enforce a timeout if a fence stops progress.
int main(int argc, char** argv) {
  try {
    const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu";
    if (argc > 1 && !cpu_only) {
      throw std::invalid_argument("only --cpu is supported");
    }
    if (!cpu_only && !gpu::is_available()) {
      throw std::runtime_error("Metal is required");
    }
    auto producer = new_stream(cpu_only ? Device::cpu : Device::gpu);
    auto host = new_stream(Device::cpu);
    constexpr int layers = 96;
    constexpr int repeats = 3;
    size_t comparisons = 0;
    for (int width : {1, 2, 3, 4, 8}) {
      for (int repeat = 0; repeat < repeats; ++repeat) {
        {
          auto x = zeros({width, 5120}, float32, producer);
          std::vector<array> roots;
          for (int layer = 0; layer < layers; ++layer) {
            auto partial = add(x, array(1.0f), producer);
            auto reduced = add(partial, array(2.0f), host);
            x = add(reduced, array(3.0f), producer);
            // Independent roots share GPU producers with the CPU chain.
            if (layer % 3 == 0) {
              roots.push_back(add(partial, array(17.0f), producer));
            }
          }
          roots.insert(roots.begin(), x);
          eval(roots);
          synchronize(producer);
          synchronize(host);
          for (size_t root = 0; root < roots.size(); ++root) {
            float expected =
                root == 0 ? layers * 6.0f : (root - 1) * 3 * 6.0f + 18.0f;
            const float* values = roots[root].data<float>();
            for (size_t i = 0; i < roots[root].size(); ++i) {
              if (values[i] != expected) {
                throw std::runtime_error("exact chain/side-root mismatch");
              }
              ++comparisons;
            }
          }
        }
        synchronize(producer);
        synchronize(host);
        clear_cache();
        if (get_active_memory() != 0) {
          throw std::runtime_error("request arrays remain after checked drain");
        }
        std::cout << "{\"event\":\"case\",\"width\":" << width
                  << ",\"repeat\":" << repeat << ",\"layers\":" << layers
                  << ",\"logical_active_bytes\":0}" << std::endl;
      }
    }
    // A second GPU stream in the same chain: each layer's partial is read by
    // the CPU and by another GPU stream, and the producer stream reads both
    // results back. The producer waits for the other GPU stream on an event
    // (a spinning wait there could starve that stream's queue).
    if (!cpu_only) {
      auto side = new_stream(Device::gpu);
      for (int width : {1, 2, 4}) {
        {
          auto x = zeros({width, 5120}, float32, producer);
          for (int layer = 0; layer < layers; ++layer) {
            auto partial = add(x, array(1.0f), producer);
            auto reduced = add(partial, array(2.0f), host);
            auto gate = add(partial, array(4.0f), side);
            // (x + 3) + (x + 5) - (x + 5) = x + 3, exact in float32.
            x = subtract(
                add(reduced, gate, producer),
                add(x, array(5.0f), producer),
                producer);
          }
          eval(x);
          synchronize(producer);
          synchronize(side);
          synchronize(host);
          const float* values = x.data<float>();
          for (size_t i = 0; i < x.size(); ++i) {
            if (values[i] != layers * 3.0f) {
              throw std::runtime_error("exact two-GPU-stream chain mismatch");
            }
            ++comparisons;
          }
        }
        synchronize(producer);
        synchronize(side);
        synchronize(host);
        std::cout << "{\"event\":\"case\",\"streams\":\"two-gpu\",\"width\":"
                  << width << ",\"layers\":" << layers << "}" << std::endl;
      }
      // Both GPU streams feed the CPU in the same chain, so no GPU stream may
      // spin on the CPU work: the producer waits for the CPU result on an
      // event of the CPU stream, and the CPU stream waits for both.
      for (int width : {1, 2, 4}) {
        {
          auto x = zeros({width, 5120}, float32, producer);
          for (int layer = 0; layer < layers; ++layer) {
            auto partial = add(x, array(1.0f), producer);
            // (x - x) + 2 = 2, exact in float32.
            auto gate = add(subtract(x, x, side), array(2.0f), side);
            auto reduced = add(partial, gate, host);
            x = add(reduced, array(3.0f), producer);
          }
          eval(x);
          synchronize(producer);
          synchronize(side);
          synchronize(host);
          const float* values = x.data<float>();
          for (size_t i = 0; i < x.size(); ++i) {
            if (values[i] != layers * 6.0f) {
              throw std::runtime_error("exact two-GPU-to-CPU chain mismatch");
            }
            ++comparisons;
          }
        }
        synchronize(producer);
        synchronize(side);
        synchronize(host);
        std::cout << "{\"event\":\"case\",\"streams\":\"two-to-cpu\",\"width\":"
                  << width << ",\"layers\":" << layers << "}" << std::endl;
      }
    }
    synchronize(producer);
    synchronize(host);
    clear_streams();
    std::cout << "{\"event\":\"passed\",\"cases\":" << (cpu_only ? 15 : 21)
              << ",\"comparisons\":" << comparisons << ",\"device\":\""
              << (cpu_only ? "cpu" : "gpu") << "\"}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
