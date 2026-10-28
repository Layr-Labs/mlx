.. _using_streams:

Using Streams
=============

.. currentmodule:: mlx.core

Specifying the :obj:`Stream`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All operations (including random number generation) take an optional
keyword argument ``stream``. The ``stream`` kwarg specifies which
:obj:`Stream` the operation should run on. If the stream is unspecified then
the operation is run on the default stream of the default device:
``mx.default_stream(mx.default_device())``.  The ``stream`` kwarg can also
be a :obj:`Device` (e.g. ``stream=my_device``) in which case the operation is
run on the default stream of the provided device
``mx.default_stream(my_device)``.

Cross-device fence ordering
---------------------------

With fast Metal synchronization, a GPU consumer can wait for work on a CPU
stream. Its wait must follow both earlier command passes and earlier kernels
in the current concurrent pass. The CPU work can depend on those GPU kernels.
``Fence::wait`` therefore inserts both dependencies before the GPU wait.

A GPU stream never spins on another GPU stream. The fast wait is a kernel that
spins until its producer's timestamp arrives, and Metal does not guarantee that
a second command queue runs while a pass of the first one spins: the producing
queue can starve and both stop. When an evaluation's graph spans two GPU
streams, a consumer on one waits for the other on a shared event, and so does a
GPU stream waiting for CPU work that can itself wait for another GPU stream.
The other CPU-to-GPU and GPU-to-CPU waits keep the fast path.

The standalone ``metal_fence_chain`` test checks exact GPU-to-CPU-to-GPU chains
with shared side roots, chains where two GPU streams feed the CPU, and
request-array retirement. CTest runs separate
processes with fast synchronization and several command-buffer sizes, with a
60-second timeout per process. This component test does not qualify distributed
collective order, a transport, or an installed model service.
