.. _unified_memory:

Unified Memory
==============

.. currentmodule:: mlx.core

Apple silicon has a unified memory architecture. The CPU and GPU have direct
access to the same memory pool. MLX is designed to take advantage of that.

Concretely, when you make an array in MLX you don't have to specify its location:


.. code-block:: python

  a = mx.random.normal((100,))
  b = mx.random.normal((100,))

Both ``a`` and ``b`` live in unified memory.

In MLX, rather than moving arrays to devices, you specify the device when you
run the operation. Any device can perform any operation on ``a`` and ``b``
without needing to move them from one memory location to another. For example:

.. code-block:: python

  mx.add(a, b, stream=mx.cpu)
  mx.add(a, b, stream=mx.gpu)

In the above, both the CPU and the GPU will perform the same add
operation. The operations can (and likely will) be run in parallel since
there are no dependencies between them. See :ref:`using_streams` for more
information the semantics of streams in MLX.

In the above ``add`` example, there are no dependencies between operations, so
there is no possibility for race conditions. If there are dependencies, the
MLX scheduler will automatically manage them. For example:

.. code-block:: python

  c = mx.add(a, b, stream=mx.cpu)
  d = mx.add(a, c, stream=mx.gpu)

In the above case, the second ``add`` runs on the GPU but it depends on the
output of the first ``add`` which is running on the CPU. MLX will
automatically insert a dependency between the two streams so that the second
``add`` only starts executing after the first is complete and ``c`` is
available.

A Simple Example
~~~~~~~~~~~~~~~~

Here is a more interesting (albeit slightly contrived example) of how unified
memory can be helpful. Suppose we have the following computation:

.. code-block:: python

  def fun(a, b, d1, d2):
    x = mx.matmul(a, b, stream=d1)
    for _ in range(500):
        b = mx.exp(b, stream=d2)
    return x, b

which we want to run with the following arguments:

.. code-block:: python

  a = mx.random.uniform(shape=(4096, 512))
  b = mx.random.uniform(shape=(512, 4))

The first ``matmul`` operation is a good fit for the GPU since it's more
compute dense. The second sequence of operations are a better fit for the CPU,
since they are very small and would probably be overhead bound on the GPU.

If we time the computation fully on the GPU, we get 2.8 milliseconds. But if we
run the computation with ``d1=mx.gpu`` and ``d2=mx.cpu``, then the time is only
about 1.4 milliseconds, about twice as fast. These times were measured on an M1
Max.

Native allocation footprint bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C++ ``get_allocation_footprint_policy()`` returns a detached value that
bounds one allocator buffer, including alignment and permitted cache reuse.
On Metal, every positive request has a minimum footprint of one VM page
before the cache-reuse allowance is added. This covers the page-backed
fallback used by the optional process-allocation cap, including requests
smaller than one page. A policy captured before cap installation remains
valid afterward. Zero bytes still has a zero bound; arithmetic overflow is
refused.

For example, with 16,384-byte VM pages, requests from 1 through 16,384 bytes
have an upper bound of 32,767 bytes. The bound is conservative; it is not a
measurement of the buffer's actual backing. The cap accounts for its fixed
heap separately. Per-buffer footprint bounds and the optional cap do not
replace whole-process memory budgets or OS headroom checks.
