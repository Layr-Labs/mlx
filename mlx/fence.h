// Copyright © 2024 Apple Inc.

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between streams.
 *
 * Calls to `wait` wait in the given stream until all previous calls to update
 * are complete on their given stream.
 *
 * The array passed to `update` is computed and visible after the call to
 * `wait` returns. The array passed to `wait` will not be read until all
 * previous calls to `update` have completed.
 *
 * Note, calls to `update` should always be from the same thread or explicitly
 * synchronized so that they occur in sequence. Calls to `wait` can be on any
 * thread.
 *
 * For the Metal back-end the fence supports slow (default) and fast mode.
 * Fast mode requires setting the environment variable
 * `MLX_METAL_FAST_SYNCH=1`. Fast mode also requires Metal 3.2+ (macOS 15+,
 * iOS 18+).
 */
class Fence {
 public:
  Fence() {};
  explicit Fence(Stream stream);
  // `fast` false synchronizes with events even when fast synchronization is
  // enabled. Backends without fast synchronization ignore it.
  Fence(Stream stream, bool fast);

  void update(Stream stream, const array& x, bool cross_device);
  void wait(Stream stream, const array& x);

  // Give the wait on input `input` of `consumer` to the consumer's own GPU
  // kernel (a fenced input of a Metal custom kernel). Returns false when it
  // cannot; call `wait` then.
  bool wait_in_consumer(Stream stream, const array& consumer, int input);

  template <typename T>
  auto& cast() const {
    return *static_cast<T*>(fence_.get());
  }

 private:
  std::shared_ptr<void> fence_;
};

} // namespace mlx::core
