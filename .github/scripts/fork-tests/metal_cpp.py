"""Fetch the pinned Apple headers before a native CI build."""

import hashlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
SHA256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
SIZE = 260405
ATTEMPTS = 4


def download(path, *, run=subprocess.run, sleep=time.sleep):
    partial = path.with_suffix(".partial")
    for attempt in range(1, ATTEMPTS + 1):
        partial.unlink(missing_ok=True)
        try:
            result = run(
                [
                    "curl",
                    "--disable",
                    "--fail",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "60",
                    "--max-filesize",
                    str(SIZE),
                    "--silent",
                    "--show-error",
                    "--output",
                    str(partial),
                    URL,
                ],
                timeout=65,
                check=False,
            )
            if result.returncode == 0 and partial.stat().st_size == SIZE:
                if hashlib.sha256(partial.read_bytes()).hexdigest() == SHA256:
                    partial.replace(path)
                    return
        except (OSError, subprocess.TimeoutExpired):
            pass
        partial.unlink(missing_ok=True)
        print(
            f"Apple header download attempt {attempt}/{ATTEMPTS} failed",
            file=sys.stderr,
        )
        if attempt < ATTEMPTS:
            sleep(2)
    raise RuntimeError("Apple headers failed the bounded download/integrity check")


def unpack(archive, destination):
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if (
            len(entries) > 512
            or sum(item.file_size for item in entries) > 16 * 1024 * 1024
        ):
            raise ValueError("Apple header archive exceeds its bounds")
        names = set()
        files = set()
        for item in entries:
            path = PurePosixPath(item.filename)
            mode = stat.S_IFMT(item.external_attr >> 16)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or "\x00" in item.filename
                or not path.parts
                or path.parts[0] != "metal-cpp"
                or path.as_posix() != item.filename.rstrip("/")
                or path in names
                or item.flag_bits & 1
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            ):
                raise ValueError("unsafe Apple header archive entry")
            names.add(path)
            if not item.is_dir():
                files.add(path)
        required = {
            PurePosixPath("metal-cpp") / name
            for name in (
                "Metal/Metal.hpp",
                "Foundation/Foundation.hpp",
                "QuartzCore/QuartzCore.hpp",
            )
        }
        if not required.issubset(files):
            raise ValueError("Apple header archive is incomplete")
        destination.mkdir()
        for item in entries:
            target = destination / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    return destination / "metal-cpp"


def main():
    directory = Path(tempfile.mkdtemp(prefix="mlx-metal-cpp-", dir="/tmp"))
    try:
        archive = directory / "headers.zip"
        download(archive)
        source = unpack(archive, directory / "source")
        print(f"cmake-args=-DFETCHCONTENT_SOURCE_DIR_METAL_CPP={source}")
    except BaseException:
        shutil.rmtree(directory)
        raise


if __name__ == "__main__":
    main()
