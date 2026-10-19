# Custom Metal launch validation.

import unittest

import mlx.core as mx


@unittest.skipIf(not mx.metal.is_available(), "Metal is unavailable; not a pass")
class ThreadgroupBoundsTests(unittest.TestCase):
    def test_invalid_dimensions_precede_compilation(self):
        # A compiler attempt would raise RuntimeError instead of ValueError.
        cases = [
            ((0, 1, 1), (1024, 1, 1), "dimensions"),
            ((1, 0, 1), (1024, 1, 1), "dimensions"),
            ((1, 1, 0), (1024, 1, 1), "dimensions"),
            ((-1, 1, 1), (1024, 1, 1), "dimensions"),
            ((1, -1, 1), (1024, 1, 1), "dimensions"),
            ((1, 1, -1), (1024, 1, 1), "dimensions"),
            ((1024, 1, 1), (-1, 1, 1), "dimensions"),
            ((1024, 1, 1), (1, -1, 1), "dimensions"),
            ((1024, 1, 1), (1, 1, -1), "dimensions"),
            ((65536, 65536, 1), (1024, 1, 1), "32 bits"),
            (
                (2147483647, 2147483647, 2147483647),
                (1024, 1, 1),
                "32 bits",
            ),
            ((2147483647, 1, 1), (1024, 1, 1), "device dimensions"),
            ((1, 2147483647, 1), (1024, 1, 1), "device dimensions"),
            ((1, 1, 2147483647), (1024, 1, 1), "device dimensions"),
        ]
        for index, (group, grid, message) in enumerate(cases):
            with self.subTest(group=group, grid=grid):
                kernel = mx.fast.metal_kernel(
                    name=f"invalid_geometry_{index}",
                    input_names=[],
                    output_names=["out"],
                    source="INTENTIONAL_INVALID_METAL;",
                    ensure_row_contiguous=True,
                )
                with self.assertRaisesRegex(ValueError, message):
                    outputs = kernel(
                        inputs=[],
                        grid=grid,
                        threadgroup=group,
                        output_shapes=[(1,)],
                        output_dtypes=[mx.uint32],
                        init_value=0,
                        stream=mx.gpu,
                    )
                    mx.eval(outputs)

    def test_legal_launch_keeps_threads_and_values(self):
        for group in [32, 256, 512, 1024]:
            kernel = mx.fast.metal_kernel(
                name="thread_count_and_math",
                input_names=["a"],
                output_names=["out", "count"],
                source="""
                    uint i = thread_position_in_grid.x;
                    out[i] = a[i] * 2.0f + 1.0f;
                    count[i] = threads_per_threadgroup.x;
                """,
            )
            a = mx.arange(2048, dtype=mx.float32)
            out, count = kernel(
                inputs=[a],
                grid=(2048, 1, 1),
                threadgroup=(group, 1, 1),
                output_shapes=[(2048,), (2048,)],
                output_dtypes=[mx.float32, mx.uint32],
                stream=mx.gpu,
            )
            mx.eval(out, count)
            self.assertTrue(mx.array_equal(out, a * 2 + 1).item())
            self.assertTrue(mx.all(count == group).item())


if __name__ == "__main__":
    unittest.main()
