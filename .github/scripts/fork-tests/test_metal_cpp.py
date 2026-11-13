import hashlib
import io
import stat
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import metal_cpp


def archive(entries=()):
    data = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(data, "w") as output:
        warnings.simplefilter("ignore", UserWarning)
        for name in (
            "Metal/Metal.hpp",
            "Foundation/Foundation.hpp",
            "QuartzCore/QuartzCore.hpp",
        ):
            output.writestr("metal-cpp/" + name, b"header")
        for name, value in entries:
            output.writestr(name, value)
    return data.getvalue()


class MetalCppIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cmake_pins_the_verified_archive(self):
        source = Path(__file__).resolve().parents[3] / "CMakeLists.txt"
        self.assertIn("URL_HASH SHA256=" + metal_cpp.SHA256, source.read_text())
        self.assertIn(metal_cpp.URL, source.read_text())

    def test_partial_transfer_retries_from_empty_file(self):
        content = archive()
        calls = []

        def run(args, **kwargs):
            path = Path(args[args.index("--output") + 1])
            self.assertFalse(path.exists())
            self.assertEqual(kwargs["timeout"], 65)
            self.assertEqual(args[args.index("--max-filesize") + 1], str(len(content)))
            self.assertIn("=https", args)
            self.assertNotIn("--insecure", args)
            calls.append(args)
            path.write_bytes(content[:5] if len(calls) == 1 else content)
            return SimpleNamespace(returncode=18 if len(calls) == 1 else 0)

        sleep = Mock()
        with (
            patch.object(metal_cpp, "SIZE", len(content)),
            patch.object(metal_cpp, "SHA256", hashlib.sha256(content).hexdigest()),
        ):
            metal_cpp.download(self.root / "headers.zip", run=run, sleep=sleep)
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root / "headers.zip").read_bytes(), content)
        sleep.assert_called_once_with(2)

    def test_timeout_clears_partial_file_before_retry(self):
        content = archive()
        calls = []

        def run(args, **kwargs):
            path = Path(args[args.index("--output") + 1])
            self.assertFalse(path.exists())
            calls.append(args)
            if len(calls) == 1:
                path.write_bytes(content[:5])
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            path.write_bytes(content)
            return SimpleNamespace(returncode=0)

        with (
            patch.object(metal_cpp, "SIZE", len(content)),
            patch.object(metal_cpp, "SHA256", hashlib.sha256(content).hexdigest()),
        ):
            metal_cpp.download(self.root / "headers.zip", run=run, sleep=Mock())
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root / "headers.zip").read_bytes(), content)

    def test_wrong_digest_aborts_after_four_attempts(self):
        content = b"bad content of expected length"
        run = Mock()

        def write(args, **kwargs):
            Path(args[args.index("--output") + 1]).write_bytes(content)
            return SimpleNamespace(returncode=0)

        run.side_effect = write
        sleep = Mock()
        with (
            patch.object(metal_cpp, "SIZE", len(content)),
            self.assertRaises(RuntimeError),
        ):
            metal_cpp.download(self.root / "headers.zip", run=run, sleep=sleep)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_transfers_abort_without_publishing(self):
        run = Mock(return_value=SimpleNamespace(returncode=18))
        with self.assertRaises(RuntimeError):
            metal_cpp.download(self.root / "headers.zip", run=run, sleep=Mock())
        self.assertEqual(run.call_count, 4)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_valid_headers_extract(self):
        path = self.root / "headers.zip"
        path.write_bytes(archive())
        result = metal_cpp.unpack(path, self.root / "source")
        self.assertEqual((result / "Metal/Metal.hpp").read_bytes(), b"header")

    def test_unsafe_paths_are_rejected_before_extraction(self):
        for name in (
            "../escape",
            "/escape",
            "metal-cpp/../escape",
            "metal-cpp\\escape",
            "other/file",
        ):
            with self.subTest(name=name):
                path = self.root / "headers.zip"
                path.write_bytes(archive([(name, b"bad")]))
                with self.assertRaises(ValueError):
                    metal_cpp.unpack(path, self.root / "source")
                self.assertFalse((self.root / "source").exists())

    def test_duplicate_or_symlink_is_rejected(self):
        link = zipfile.ZipInfo("metal-cpp/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for extra in [
            [("metal-cpp/Metal/Metal.hpp", b"duplicate")],
            [(link, b"/escape")],
        ]:
            path = self.root / "headers.zip"
            path.write_bytes(archive(extra))
            with self.assertRaises(ValueError):
                metal_cpp.unpack(path, self.root / "source")
            self.assertFalse((self.root / "source").exists())

    def test_expanded_archive_bound_is_enforced(self):
        path = self.root / "headers.zip"
        path.write_bytes(archive([("metal-cpp/large", b"x" * (16 * 1024 * 1024))]))
        with self.assertRaises(ValueError):
            metal_cpp.unpack(path, self.root / "source")
        self.assertFalse((self.root / "source").exists())


if __name__ == "__main__":
    unittest.main()
