.. _environment_variables:

Environment Variables
=====================

MLX uses environment variables to configure compilation, numerical precision,
backend behavior, and distributed execution. Set them before starting the
process. Many variables are read when the corresponding subsystem is first
initialized and changing them later may have no effect.

Boolean variables use ``0`` to disable and a nonzero integer to enable unless
otherwise noted.

General
-------

.. envvar:: MLX_DISABLE_COMPILE

   Disable compilation globally. This variable is enabled by its presence, so
   setting it to ``0`` also disables compilation. Calling
   :func:`mlx.core.enable_compile` overrides it.

.. envvar:: MLX_ENABLE_TF32

   Allow reduced-precision ``float32`` matrix-multiplication family operations
   on supported hardware. The default is ``1``. Set it to ``0`` to keep these
   operations in full ``float32`` precision. See :doc:`precision`.

Distributed
-----------

The variables required to initialize a distributed process depend on the
backend. :doc:`distributed` describes their formats and how ``mlx.launch``
sets them.

.. envvar:: MLX_RANK

   The zero-based rank of the current process. This is used by the Ring,
   JACCL, and NCCL backends. ``JACCL_RANK`` is accepted as a higher-priority
   alias by JACCL.

.. envvar:: MLX_HOSTFILE

   The path to the JSON host file used by the Ring backend.

.. envvar:: MLX_RING_VERBOSE

   Enable verbose logging for the Ring backend. This variable is enabled by
   its presence.

.. envvar:: MLX_IBV_DEVICES

   The path to the JSON device-connectivity file used by JACCL.
   ``JACCL_IBV_DEVICES`` is accepted as a higher-priority alias.

.. envvar:: MLX_JACCL_COORDINATOR

   The coordinator address in ``IP:port`` form used to establish JACCL
   connections. ``JACCL_COORDINATOR`` is accepted as a higher-priority alias.

.. envvar:: MLX_JACCL_RING

   Prefer a ring topology for JACCL. This variable is enabled by its presence.
   ``JACCL_RING`` is accepted as a higher-priority alias.

.. envvar:: MLX_JACCL_PROGRESS_TIMEOUT_MS

   The longest time in milliseconds that a JACCL call waits for an RDMA
   completion. The default is ``30000``. Set it to ``0`` or a negative value
   to remove the limit. ``JACCL_PROGRESS_TIMEOUT_MS`` is accepted as a
   higher-priority alias. When no completion arrives for this time, the call
   fails and the group is closed: its queue pairs and completion queues are
   destroyed and its buffers are deregistered. What follows is set by
   :envvar:`MLX_JACCL_TIMEOUT_ACTION`. See the :ref:`JACCL section
   <jaccl_section>`.

.. envvar:: MLX_JACCL_TIMEOUT_ACTION

   What a JACCL call does after it failed (no completion within
   :envvar:`MLX_JACCL_PROGRESS_TIMEOUT_MS`, a failed completion, or a failed
   poll) and closed its group. ``teardown-exit``, the default, sets the wired
   limit to zero, clears the buffer cache and exits the process with status 75
   without running static destructors. ``throw`` raises an error; each later
   call on the group then raises an error immediately. Any other value keeps
   the default. ``JACCL_TIMEOUT_ACTION`` is accepted as a higher-priority
   alias.

.. envvar:: MLX_WORLD_SIZE

   The total number of processes in an NCCL group.

.. envvar:: MLX_NCCL_TIMEOUT

   The timeout in milliseconds for establishing NCCL bootstrap connections.
   The default is ``300000``.

.. envvar:: MLX_MPI_LIBNAME

   Override the MPI dynamic-library name. The default is ``libmpi.dylib`` on
   macOS and ``libmpi.so`` on other platforms.

The NCCL backend also requires ``NCCL_HOST_IP`` and ``NCCL_PORT``. Setting
``NCCL_DEBUG=INFO`` enables additional logging while MLX establishes the
bootstrap connection. NCCL itself recognizes additional `NCCL environment
variables <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>`_.
``CUDA_VISIBLE_DEVICES`` selects the local CUDA device for each process and is
handled by the CUDA runtime.

Metal
-----

.. envvar:: MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS

   Re-request residency for existing wired allocations at this interval in
   milliseconds. This works around macOS releasing residency while the GPU is
   idle. The default is ``0`` (disabled); nonpositive values disable renewal.
   For example, ``1000`` renews once per second. This does not change the wired
   limit set by :func:`mlx.core.set_wired_limit` or submit GPU work. Renewal
   continues while the Metal device exists and can keep memory wired during
   long idle periods. Requires a Metal 3 GPU on macOS 15 or iOS 18 or later.

.. envvar:: MLX_METAL_FAST_SYNCH

   Enable the faster Metal CPU/GPU synchronization path. The default is ``0``.
   This requires Metal 3.2 or later (macOS 15 or later, or iOS 18 or later).

.. envvar:: MLX_METAL_FUSED_FENCE_UPDATE

   With ``MLX_METAL_FAST_SYNCH=1``, make a GPU array visible to the CPU with
   one dispatch instead of two: the coherent rewrite of the array and the
   fence timestamp. The default is ``0``.

.. envvar:: MLX_METAL_FENCED_INPUTS

   With ``MLX_METAL_FAST_SYNCH=1``, a custom kernel made with
   ``metal_kernel_with_fenced_inputs`` waits itself for a fenced input that a
   CPU stream makes in the same evaluation, and MLX does not encode a separate
   fence wait for that input. The default is ``0``.

.. envvar:: MLX_CONV_WINOGRAD

   Allow the Winograd algorithm for 3x3 ``conv2d``. The default is ``1``. Set
   it to ``0`` to use the GEMM convolution for all inputs. See
   :doc:`precision`.

.. envvar:: MLX_QMV_BATCH_INVARIANT

   Set it to ``1`` to compute a quantized matrix product ``x @ w.T`` with
   fewer rows than the vector limit only with the vector kernels: ``qmv`` for
   one row and ``qmv_wide`` from two rows. Then each row gets the same bits at
   each row count from 2 to the limit - 1, so the rows of different requests
   can share one product. The default, ``0``, also uses faster kernels that
   give a row different bits at some row counts: ``qmv_mma`` for 6 to 16 rows
   of the floating-point modes, and the affine split-K kernel from 8 rows on
   GPUs with neural accelerators. The vector limit is 6 to 33 rows; it
   depends on the GPU and on the shape. A row alone (``qmv``) can get bits
   that are different from the bits of the same row among two or more rows.
   MLX reads this variable once in each process.

Advanced tuning
---------------

These variables tune MLX implementation details. They are primarily intended
for development, diagnostics, and performance experiments. The defaults are
selected automatically for the current hardware and are appropriate for most
users. Their behavior may change as the implementation evolves.

.. envvar:: MLX_BFS_MAX_WIDTH

   Set the breadth-first-search width limit used when constructing an
   evaluation tape. The default is ``20``.

.. envvar:: MLX_MAX_OPS_PER_BUFFER

   Override the maximum number of operations encoded in one Metal command
   buffer or CUDA graph. The default depends on the device.

.. envvar:: MLX_MAX_MB_PER_BUFFER

   Override the approximate memory limit, in megabytes, for one Metal command
   buffer or CUDA graph. The default depends on the device.

.. envvar:: MLX_METAL_GPU_ARCH

   Override the Metal GPU architecture string reported to MLX. This affects
   architecture-specific kernel and scheduling choices, but does not change
   the capabilities of the physical GPU. Forcing an architecture that does not
   match the GPU can select incompatible kernels and produce incorrect results.

.. envvar:: MLX_SDPA_BLOCKS

   Override the number of reduction blocks used by the Metal scaled
   dot-product attention kernel. Positive values are rounded up to a multiple
   of ``32``.

.. envvar:: MLX_QMM_STEEL_TILE

   Output tile of the Metal steel affine quantized matmul (``x @ w.T`` with
   at least 64 rows and a multiple of 64 outputs) where no neural-accelerator
   kernel serves the product. The default, ``0``, takes 64 rows x 32 columns
   for 16-bit inputs and 32 x 64 for FP32 inputs from 256 rows, and 32 x 32
   below. ``32`` (32 x 32, the previous tile), ``64`` (64 x 64), ``6432``
   (64 rows x 32 columns), ``3264`` (32 x 64) or ``6464`` (64 x 64 at a K
   depth of 64 for 16-bit inputs in groups of 64 or more) choose one tile for
   every product from 64 rows. Every tile gives the same bits; only the speed
   changes. Only builds that compile kernels at run time (``MLX_METAL_JIT=ON``)
   take a tile other than 32 x 32: the precompiled kernel library of the
   default build holds the 32 x 32 kernels only, so there this variable has no
   effect.

.. envvar:: MLX_QMM_STEEL_TILE_MIN_GROUPS

   Fewest 64 x 64 threadgroups a product must fill to take a tile other than
   32 x 32; smaller products keep 32 x 32. The default is ``0``.

CUDA
----

The MLX-prefixed variables in this section are advanced CUDA backend controls.

.. envvar:: MLX_USE_CUDA_GRAPHS

   Enable CUDA graph capture and replay. The default is ``1``.

.. envvar:: MLX_SAVE_CUDA_GRAPHS_DOT_FILE

   Use the specified value as the filename prefix when writing captured CUDA
   graphs to numbered DOT files. An unset or empty value disables the output.

.. envvar:: MLX_PTX_CACHE_DIR

   Override the directory used to cache runtime-compiled PTX. By default MLX
   uses an ``mlx/<version>/ptx`` directory under the system temporary
   directory.

.. envvar:: MLX_CUDA_USE_CUDNN_SDPA

   Allow the CUDA backend to use cuDNN scaled dot-product attention when the
   inputs and device are supported. The default is ``1``.

.. envvar:: MLX_CUDA_CONV_CACHE_SIZE

   Set the CUDA convolution cache capacity. The default is ``128``.

.. envvar:: MLX_CUDA_FFT_CACHE_SIZE

   Set the CUDA FFT plan cache capacity. The default is ``128``.

.. envvar:: MLX_CUDA_GRAPH_CACHE_SIZE

   Set the CUDA graph cache capacity. The default is ``400``.

.. envvar:: MLX_CUDA_SDPA_CACHE_SIZE

   Set the CUDA forward scaled dot-product attention cache capacity. The
   default is ``256``.

.. envvar:: MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE

   Set the CUDA backward scaled dot-product attention cache capacity. The
   default is ``64``.

.. envvar:: MLX_ENABLE_CACHE_THRASHING_CHECK

   Detect consecutive CUDA cache misses (more than twice the capacity with no
   hit in between) and raise an error suggesting a larger cache capacity. The
   default is ``1``.

MLX also uses ``CUDA_HOME`` or ``CUDA_PATH`` to locate CUDA headers for runtime
kernel compilation when they cannot be found in the Python environment.

CPU
---

.. envvar:: MLX_CPU_CACHE_DIR

   Override the directory used to cache runtime-compiled CPU kernels. An unset
   or empty value uses ``mlx/<version>/cpu`` under the system temporary
   directory, which is the default.
