#!/usr/bin/env python3
"""Run MLX's Fedora 42 C++ build test in Fedora userspace on a Darkbloom ARM64 runner.

The sandbox has no Docker, so the unchanged Fedora build script
(.github/scripts/setup+build-cpp-linux-fedora-container.sh) runs under PRoot, built here
from pinned source, in the official archived Fedora 42 container rootfs. PRoot
translates paths and fakes UID 0 through ptrace; it is not an isolation boundary (the
job VM is).
"""

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

PTRACE_PROBE = r"""
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
  if (getuid() != 1000 || geteuid() != 1000 || prctl(PR_GET_NO_NEW_PRIVS,0,0,0,0) != 1)
    return 10;
  printf("PTRACE_PROBE uid=%lu dumpable=%d no_new_privs=1\n",
         (unsigned long)getuid(), prctl(PR_GET_DUMPABLE));
  fflush(stdout);
  pid_t child = fork();
  if (child < 0) { perror("fork"); return 11; }
  if (!child) {
    if (ptrace(PTRACE_TRACEME,0,0,0)) { perror("PTRACE_TRACEME"); _exit(12); }
    raise(SIGSTOP); _exit(0);
  }
  int status;
  if (waitpid(child,&status,0) != child || !WIFSTOPPED(status)) return 13;
  if (ptrace(PTRACE_CONT,child,0,0)) { perror("PTRACE_CONT"); return 14; }
  if (waitpid(child,&status,0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
    return 15;
  puts("PTRACE_PROBE_OK"); return 0;
}
"""


def load_pins():
    return json.loads(Path(__file__).with_name("pins.json").read_text())


def verified(path, spec):
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"wrong size: {path.name}")
    with path.open("rb") as src:
        digest = hashlib.file_digest(src, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"wrong SHA-256: {path.name}")


OCI_GZIP_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"


def oci_layer(archive, spec, target):
    """Write the one pinned gzip layer of the pinned archived OCI image to target.

    Only index.json, the pinned manifest and the pinned layer are read; the index
    must name exactly that manifest, whose bytes must hash to its digest, and the
    manifest must list exactly that one layer (media type, digest and size).
    """
    if target.exists():
        verified(target, spec)
        return
    manifest_sha = spec["manifest"].removeprefix("sha256:")
    part = target.with_name(target.name + ".part")
    with tarfile.open(archive, "r:xz") as src:

        def regular(name, limit):
            member = src.getmember(name)
            if not member.isfile() or member.size > limit:
                raise ValueError(f"unexpected OCI member: {name}")
            return member

        index = json.loads(src.extractfile(regular("index.json", 1 << 16)).read())
        manifests = index.get("manifests")
        if (
            not isinstance(manifests, list)
            or len(manifests) != 1
            or manifests[0].get("digest") != spec["manifest"]
        ):
            raise ValueError("OCI index must name exactly the pinned manifest")
        raw = src.extractfile(regular("blobs/sha256/" + manifest_sha, 1 << 16)).read()
        if hashlib.sha256(raw).hexdigest() != manifest_sha:
            raise ValueError("OCI manifest does not match its digest")
        layers = json.loads(raw).get("layers")
        if (
            not isinstance(layers, list)
            or len(layers) != 1
            or layers[0].get("mediaType") != OCI_GZIP_LAYER
            or layers[0].get("digest") != "sha256:" + spec["sha256"]
            or layers[0].get("size") != spec["size"]
        ):
            raise ValueError("OCI manifest must list exactly the pinned gzip layer")
        with (
            src.extractfile(
                regular("blobs/sha256/" + spec["sha256"], spec["size"])
            ) as data,
            part.open("xb") as out,
        ):
            shutil.copyfileobj(data, out)
    verified(part, spec)
    part.rename(target)


def patch_proot_faccessat2(src):
    """Add faccessat2 (439) to PRoot 5.4.1's ARM64 syscall table before it is built.

    The table stops at clone3, so PRoot passes faccessat2 to the kernel untranslated:
    a guest `test -x` (bash uses faccessat2) would read the host's path, not the
    guest's. enter.c and the seccomp filter already handle PR_faccessat2.
    """
    table = src / "syscall" / "sysnums-arm64.h"
    text = table.read_text()
    anchor = "    [435] = PR_clone3,\n"
    if text.count(anchor) != 1 or "PR_faccessat2" in text:
        raise RuntimeError("unexpected PRoot ARM64 syscall table")
    text = text.replace(anchor, anchor + "    [439] = PR_faccessat2,\n")
    table.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_inputs(inputs, pins):
    """Every pinned download, and the rootfs layer derived from the OCI archive."""
    for name, spec in pins["files"].items():
        verified(inputs / name, spec)
    verified(inputs / "fedora.tar.gz", pins["fedora_layer"])


def download(url, part, size, attempts=3, pause=5):
    """One pinned download with bounded retries; any stale partial file goes first."""
    for attempt in range(1, attempts + 1):
        part.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as src, part.open("xb") as out:
                copied = 0
                while chunk := src.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > size:
                        raise ValueError(f"oversized download: {part.name}")
                    out.write(chunk)
            if copied != size:
                raise ValueError(
                    f"truncated download: {part.name} ({copied} of {size} bytes)"
                )
            return
        except (OSError, ValueError, http.client.HTTPException) as error:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(
                f"FEDORA_PROOT_RETRY {part.name} attempt {attempt}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(pause * attempt)


def fetch_inputs(dest, pins):
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in pins["files"].items():
        target = dest / name
        if not target.exists():
            part = dest / (name + ".part")
            download(spec["url"], part, spec["size"])
            verified(part, spec)
            part.rename(target)
        verified(target, spec)
    layer = pins["fedora_layer"]
    oci_layer(dest / layer["archive"], layer, dest / "fedora.tar.gz")


def extract_root(archive, dest):
    """One pinned layer, no overlay merging; links are created after regular files.

    This order prevents tar members writing through links, including absolute
    rootfs links. Device nodes are unnecessary: /dev is an explicit PRoot bind.
    All real ownership remains the calling runner; setuid/setgid bits are dropped.
    """
    dest.mkdir()
    dest = dest.resolve()
    seen = set()
    links = []
    with tarfile.open(archive, "r:*") as src:
        for member in src:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or not p.parts:
                raise ValueError(f"unsafe tar path: {member.name}")
            if any(part.startswith(".wh.") for part in p.parts):
                raise ValueError("overlay whiteouts require a different unpacker")
            key = str(p)
            if key == ".":
                continue
            if key in seen:
                raise ValueError(f"duplicate tar path: {key}")
            seen.add(key)
            target = dest.joinpath(*p.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with src.extractfile(member) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out)
                target.chmod(member.mode & 0o777)
            elif member.issym() or member.islnk():
                links.append((member, target))
            elif not (member.ischr() or member.isblk() or member.isfifo()):
                raise ValueError(f"unsupported tar entry: {key}")
        # Hardlinks must name an already extracted regular file, never a symlink.
        for member, target in links:
            if not member.islnk():
                continue
            link = PurePosixPath(member.linkname)
            source = dest.joinpath(*link.parts)
            if link.is_absolute() or ".." in link.parts or not source.is_file():
                raise ValueError(f"unsafe hardlink: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        for member, target in links:
            if member.issym():
                if not target.parent.resolve().is_relative_to(dest):
                    raise ValueError(f"symlink parent escapes root: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)


def archived_repos(root):
    root = root.resolve(strict=True)
    repo_dir = root / "etc/yum.repos.d"
    target = repo_dir / "pilot-fedora42.repo"
    if root.stat().st_uid != os.geteuid():
        raise ValueError("rootfs is not owned by this job user")
    for path in (repo_dir, target):
        if not path.resolve().is_relative_to(root):
            raise ValueError("repository configuration escapes the owned rootfs")
    repo_dir.mkdir(parents=True, exist_ok=True)
    for repo in repo_dir.glob("*.repo"):
        repo.rename(repo.with_suffix(".repo.disabled-pilot"))
    key = "file:///etc/pki/rpm-gpg/RPM-GPG-KEY-fedora-42-primary"
    text = ""
    for name, suffix in (
        ("fedora", "releases/42/Everything/aarch64/os"),
        ("updates", "updates/42/Everything/aarch64"),
    ):
        text += (
            f"[{name}]\nname=Fedora 42 archived {name}\n"
            f"baseurl=https://archives.fedoraproject.org/pub/archive/fedora/linux/{suffix}/\n"
            f"enabled=1\ngpgcheck=1\ngpgkey={key}\n\n"
        )
    target.write_text(text)


SCRIPT = ".github/scripts/setup+build-cpp-linux-fedora-container.sh"


def run_pilot(work, source, pins):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("run only on a Linux ARM64 runner")
    if os.getuid() != 1000 or os.geteuid() != 1000:
        raise RuntimeError("run as the unchanged UID1000 job user")
    status = Path("/proc/self/status").read_text()
    if "NoNewPrivs:\t1" not in status:
        raise RuntimeError("the job must inherit no_new_privs")
    work.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    evidence = {
        "pins": pins,
        "host_arch": platform.machine(),
        "uid": os.getuid(),
        "kernel": platform.release(),
        "steps": [],
        "result": "running",
    }

    def call(name, args, *, cwd=work, env=None, timeout=180):
        before = time.monotonic()
        print(f"FEDORA_PROOT_STEP {name}", flush=True)
        with (work / (name + ".log")).open("w") as log:
            proc = subprocess.Popen(
                args,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                print(
                    (work / (name + ".log")).read_text(errors="replace"),
                    end="",
                    flush=True,
                )
                raise
        duration = time.monotonic() - before
        evidence["steps"].append(
            {"name": name, "seconds": duration, "exit": proc.returncode}
        )
        print((work / (name + ".log")).read_text(errors="replace"), end="", flush=True)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, args)

    try:
        script = (source / SCRIPT).resolve(strict=True)
        if not script.is_file() or not script.is_relative_to(source):
            raise RuntimeError("the checked-out Fedora build script is missing")
        inputs = work / "inputs"
        fetch_inputs(inputs, pins)
        verify_inputs(inputs, pins)
        (work / "ptrace.c").write_text(PTRACE_PROBE)
        call(
            "ptrace-build",
            ["cc", "-O2", "-Wall", "-Wextra", "-Werror", "ptrace.c", "-o", "ptrace"],
        )
        call("ptrace", [str(work / "ptrace")], timeout=15)
        deps = work / "deps"
        for deb in (
            "libtalloc-dev.deb",
            "libtalloc2.deb",
            "pkgconf-bin.deb",
            "libpkgconf3.deb",
        ):
            call(deb, ["dpkg-deb", "--extract", str(inputs / deb), str(deps)])
        (deps / "usr/bin/pkg-config").symlink_to("pkgconf")
        extract_root(inputs / "proot.tar.gz", work / "proot-source")
        proot_src = work / "proot-source" / ("proot-" + pins["proot_commit"]) / "src"
        evidence["proot_arm64_table_sha256"] = patch_proot_faccessat2(proot_src)
        lib = deps / "usr/lib/aarch64-linux-gnu"
        env = dict(
            os.environ,
            PKG_CONFIG_LIBDIR=str(lib / "pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(deps),
            LDFLAGS=f"-Wl,-rpath,{lib}",
            PATH=f"{deps / 'usr/bin'}:{os.environ['PATH']}",
            LD_LIBRARY_PATH=str(lib),
        )
        call(
            "proot-build",
            ["make", "-j2", "HAS_SWIG=", "HAS_PYTHON_CONFIG=", "proot"],
            cwd=proot_src,
            env=env,
        )
        proot = proot_src / "proot"
        call("proot-version", [str(proot), "--version"])
        with proot.open("rb") as src:
            evidence["proot_sha256"] = hashlib.file_digest(src, "sha256").hexdigest()
        extract_root(inputs / "fedora.tar.gz", work / "fedora")
        root = work / "fedora"
        if "VERSION_ID=42" not in (root / "etc/os-release").read_text():
            raise RuntimeError("rootfs is not Fedora42")
        archived_repos(root)
        scratch = work / "tmp"
        scratch.mkdir()
        command = [
            str(proot),
            "-0",
            "-r",
            str(root),
            "-b",
            "/dev",
            "-b",
            "/proc",
            "-b",
            f"{scratch}:/tmp",
            "-b",
            f"{source}:/work",
            "-w",
            "/work",
        ]
        # The checkout belongs to the runner's UID 1000; git in the fake-root guest
        # must accept it.
        guest = dict(
            os.environ,
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="safe.directory",
            GIT_CONFIG_VALUE_0="/work",
        )
        # No sudo, no Docker, no namespaces/capabilities, no host /etc or /run bind.
        call(
            "fedora-smoke",
            command
            + [
                "/bin/bash",
                "-ec",
                "cat /etc/fedora-release; test $(uname -m) = aarch64; "
                # faccessat2 translation: Fedora has dnf and no apt-get; the Ubuntu host
                # has apt-get.
                "test -x /usr/bin/dnf; test ! -x /usr/bin/apt-get; "
                "test $(id -u) = 0; ldd --version; dnf --version; "
                "grep '^NoNewPrivs:' /proc/self/status; (true)& wait",
            ],
            env=guest,
            timeout=60,
        )
        # Exercise actual DNS/proxy/TLS and both literal archive repositories
        # before the expensive build, using Fedora's own libcurl/trust store.
        call(
            "fedora-archive-network",
            command + ["/usr/bin/dnf", "-q", "makecache", "--refresh"],
            env=guest,
            timeout=300,
        )
        call(
            "fedora-build-test",
            command + ["/bin/bash", "/work/" + SCRIPT],
            env=guest,
            timeout=1800,
        )
        evidence["result"] = "pass"
    except BaseException as error:
        evidence["result"] = "failure"
        evidence["error_type"] = type(error).__name__
        raise
    finally:
        evidence["seconds"] = time.monotonic() - start
        (work / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in evidence.items() if k != "pins"}, indent=2),
            flush=True,
        )
        print("FEDORA_PROOT_EVIDENCE " + str(work / "evidence.json"), flush=True)
    print("FEDORA_PROOT_OK " + evidence["result"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "run"))
    parser.add_argument(
        "--inputs", type=Path, help="fetch: download and verify the pinned inputs here"
    )
    parser.add_argument(
        "--source", type=Path, help="run: the MLX checkout to build and test"
    )
    parser.add_argument("--work", type=Path, help="run: a new private work directory")
    args = parser.parse_args()
    pins = load_pins()
    if args.action == "fetch":
        if args.inputs is None:
            parser.error("fetch requires --inputs")
        fetch_inputs(args.inputs.resolve(), pins)
    else:
        if args.source is None or args.work is None:
            parser.error("run requires --source and --work")
        run_pilot(args.work.resolve(), args.source.resolve(strict=True), pins)


if __name__ == "__main__":
    main()
