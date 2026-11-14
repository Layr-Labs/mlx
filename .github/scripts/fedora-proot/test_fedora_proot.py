import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pilot", HERE / "fedora_proot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class InputValidation(unittest.TestCase):
    def test_size_and_digest_must_both_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input"
            p.write_bytes(b"approved")
            spec = {"size": 8, "sha256": hashlib.sha256(b"approved").hexdigest()}
            pilot.verified(p, spec)
            p.write_bytes(b"replaced")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pilot.verified(p, spec)
            p.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "size"):
                pilot.verified(p, spec)

    def extract(self, entries, tmp):
        archive = Path(tmp) / "layer.tar"
        with tarfile.open(archive, "w") as tar:
            for name, typ, content in entries:
                info = tarfile.TarInfo(name)
                info.type = typ
                if typ == tarfile.REGTYPE:
                    info.size = len(content)
                    info.mode = 0o6755
                    tar.addfile(info, io.BytesIO(content))
                else:
                    info.linkname = content
                    tar.addfile(info)
        root = Path(tmp) / "root"
        pilot.extract_root(archive, root)
        return root

    def test_absolute_rootfs_symlink_is_created_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.extract(
                [
                    ("lib", tarfile.SYMTYPE, "/usr/lib"),
                    ("usr/lib/tool", tarfile.REGTYPE, b"tool"),
                ],
                tmp,
            )
            self.assertEqual((root / "lib").readlink(), Path("/usr/lib"))
            self.assertEqual((root / "usr/lib/tool").read_bytes(), b"tool")
            self.assertEqual((root / "usr/lib/tool").stat().st_mode & 0o6000, 0)

    def test_bad_paths_and_overlay_layers_are_refused(self):
        for name in ("/outside", "../outside", "usr/../../outside", "usr/.wh.hidden"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.extract([(name, tarfile.REGTYPE, b"bad")], tmp)

    def test_symlink_cannot_redirect_another_symlink_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "parent escapes"):
                self.extract(
                    [
                        ("escape", tarfile.SYMTYPE, tmp),
                        ("escape/other", tarfile.SYMTYPE, "/usr/lib"),
                    ],
                    tmp,
                )
            self.assertFalse((Path(tmp) / "other").exists())

    def test_duplicate_and_escaping_hardlinks_are_refused(self):
        cases = [
            [("tool", tarfile.REGTYPE, b"x"), ("tool", tarfile.REGTYPE, b"y")],
            [("tool", tarfile.LNKTYPE, "../../outside")],
        ]
        for entries in cases:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                self.extract(entries, tmp)

    def test_archived_repos_keep_package_signature_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "etc/yum.repos.d"
            repo.mkdir(parents=True)
            (repo / "mirror.repo").write_text("metalink=https://unbounded.example/\n")
            pilot.archived_repos(root)
            files = list(repo.glob("*.repo"))
            self.assertEqual(len(files), 1)
            text = files[0].read_text()
            self.assertEqual(text.count("gpgcheck=1"), 2)
            self.assertNotIn("metalink", text)
            self.assertEqual(text.count("https://archives.fedoraproject.org/"), 2)

    def test_repository_writes_cannot_follow_rootfs_links_outside(self):
        for escape_file in (False, True):
            with (
                self.subTest(escape_file=escape_file),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp) / "root"
                outside = Path(tmp) / "outside"
                root.mkdir()
                outside.mkdir()
                (root / "etc").mkdir()
                if escape_file:
                    (root / "etc/yum.repos.d").mkdir()
                    (root / "etc/yum.repos.d/pilot-fedora42.repo").symlink_to(
                        outside / "target"
                    )
                else:
                    (root / "etc/yum.repos.d").symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    pilot.archived_repos(root)
                self.assertEqual(list(outside.iterdir()), [])


class OciLayer(unittest.TestCase):
    """The archived Fedora OCI image: only its one pinned gzip layer, by digest."""

    LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"

    def build(self, tmp, layers=1, manifests=1, media=None, tamper_manifest=False):
        layer = b"layer-bytes-" * 100
        digest = hashlib.sha256(layer).hexdigest()
        entries = [
            {
                "mediaType": media or self.LAYER,
                "digest": "sha256:" + digest,
                "size": len(layer),
            }
        ] * layers
        manifest = json.dumps({"schemaVersion": 2, "layers": entries}).encode()
        mdigest = hashlib.sha256(manifest).hexdigest()
        stored = manifest + (b" " if tamper_manifest else b"")
        index = json.dumps(
            {
                "schemaVersion": 2,
                "manifests": [{"digest": "sha256:" + mdigest}] * manifests,
            }
        ).encode()
        archive = Path(tmp) / "fedora.oci.tar.xz"
        with tarfile.open(archive, "w:xz") as tar:
            for name, data in (
                ("index.json", index),
                ("blobs/sha256/" + mdigest, stored),
                ("blobs/sha256/" + digest, layer),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        spec = {
            "archive": "fedora.oci.tar.xz",
            "manifest": "sha256:" + mdigest,
            "sha256": digest,
            "size": len(layer),
        }
        return archive, spec, layer

    def test_pinned_layer_is_extracted_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, spec, layer = self.build(tmp)
            target = Path(tmp) / "fedora.tar.gz"
            pilot.oci_layer(archive, spec, target)
            self.assertEqual(target.read_bytes(), layer)
            pilot.oci_layer(archive, spec, target)  # an existing verified layer is kept

    def test_other_manifest_layers_or_media_are_refused(self):
        cases = [
            dict(layers=2),
            dict(manifests=2),
            dict(media="application/vnd.oci.image.layer.v1.tar"),
            dict(tamper_manifest=True),
        ]
        for case in cases:
            with tempfile.TemporaryDirectory() as tmp:
                archive, spec, _ = self.build(tmp, **case)
                with self.assertRaises(ValueError):
                    pilot.oci_layer(archive, spec, Path(tmp) / "fedora.tar.gz")
                self.assertFalse((Path(tmp) / "fedora.tar.gz").exists())

    def test_pins_must_name_the_manifest_and_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, spec, _ = self.build(tmp)
            for change in (
                {"manifest": "sha256:" + "0" * 64},
                {"sha256": "0" * 64},
                {"size": spec["size"] - 1},
            ):
                with self.assertRaises(ValueError):
                    pilot.oci_layer(
                        archive, dict(spec, **change), Path(tmp) / "fedora.tar.gz"
                    )
            self.assertFalse((Path(tmp) / "fedora.tar.gz").exists())

    def rebuild(self, tmp, archive, replace):
        """Copy archive, replacing members by name: (TarInfo, bytes or None)."""
        out = Path(tmp) / "hostile.oci.tar.xz"
        with tarfile.open(archive, "r:xz") as src, tarfile.open(out, "w:xz") as dst:
            for member in src:
                if member.name in replace:
                    info, data = replace[member.name]
                    dst.addfile(info, io.BytesIO(data) if data is not None else None)
                else:
                    dst.addfile(member, src.extractfile(member))
        return out

    def test_non_regular_or_oversized_members_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, spec, _ = self.build(tmp)
            link = tarfile.TarInfo("blobs/sha256/" + spec["sha256"])
            link.type = tarfile.SYMTYPE
            link.linkname = "index.json"
            big = json.dumps(
                {"manifests": [{"digest": spec["manifest"]}], "pad": "x" * (1 << 16)}
            ).encode()
            index = tarfile.TarInfo("index.json")
            index.size = len(big)
            for replace in ({link.name: (link, None)}, {"index.json": (index, big)}):
                hostile = self.rebuild(tmp, archive, replace)
                with self.assertRaisesRegex(ValueError, "unexpected OCI member"):
                    pilot.oci_layer(hostile, spec, Path(tmp) / "fedora.tar.gz")
                self.assertFalse((Path(tmp) / "fedora.tar.gz").exists())

    def test_existing_target_must_be_the_pinned_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, spec, _ = self.build(tmp)
            target = Path(tmp) / "fedora.tar.gz"
            target.write_bytes(b"not the layer")
            with self.assertRaises(ValueError):
                pilot.oci_layer(archive, spec, target)

    def test_fetch_inputs_derives_the_layer_and_verify_inputs_checks_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, spec, layer = self.build(tmp)
            raw = archive.read_bytes()
            pins = {
                "files": {
                    "fedora.oci.tar.xz": {
                        "url": "https://unused.invalid/",
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
                "fedora_layer": spec,
            }
            dest = Path(tmp) / "inputs"
            dest.mkdir()
            (dest / "fedora.oci.tar.xz").write_bytes(raw)
            pilot.fetch_inputs(dest, pins)  # present and verified: no download
            self.assertEqual((dest / "fedora.tar.gz").read_bytes(), layer)
            pilot.verify_inputs(dest, pins)
            (dest / "fedora.tar.gz").write_bytes(b"x" * len(layer))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pilot.verify_inputs(dest, pins)

    def test_rootfs_comes_from_the_archived_oci_image(self):
        pins = json.loads((HERE / "pins.json").read_text())
        layer = pins["fedora_layer"]
        self.assertNotIn("fedora.tar.gz", pins["files"])
        source = pins["files"][layer["archive"]]
        self.assertTrue(
            source["url"].startswith(
                "https://archives.fedoraproject.org/pub/archive/fedora/linux/releases/42/Container/aarch64/images/"
            )
        )
        self.assertEqual(pins["fedora_manifest"], layer["manifest"])
        self.assertEqual(len(layer["sha256"]), 64)
        self.assertGreater(layer["size"], 0)


class ProotFaccessat2(unittest.TestCase):
    """PRoot 5.4.1's ARM64 table lacks faccessat2 (439).

    Guest access checks would otherwise read host paths.
    """

    TABLE = (
        "static const Sysnum sysnums_arm64[] = {\n"
        "    [291] = PR_statx,\n"
        "    [435] = PR_clone3,\n"
        "};\n"
    )

    def table(self, tmp, text):
        path = Path(tmp) / "syscall"
        path.mkdir()
        (path / "sysnums-arm64.h").write_text(text)
        return Path(tmp)

    def test_faccessat2_is_added_once_after_clone3(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self.table(tmp, self.TABLE)
            digest = pilot.patch_proot_faccessat2(src)
            text = (src / "syscall/sysnums-arm64.h").read_text()
            self.assertIn("    [435] = PR_clone3,\n    [439] = PR_faccessat2,\n", text)
            self.assertEqual(digest, hashlib.sha256(text.encode()).hexdigest())

    def test_unexpected_tables_are_refused(self):
        for text in (
            self.TABLE.replace("[435] = PR_clone3,", ""),
            self.TABLE + "    [435] = PR_clone3,\n",
            self.TABLE.replace("};", "    [439] = PR_faccessat2,\n};"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    pilot.patch_proot_faccessat2(self.table(tmp, text))


class Downloads(unittest.TestCase):
    """Pinned downloads retry a bounded number of times; no partial file is kept."""

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_a_stale_partial_file_is_removed_before_the_first_attempt(self):
        calls = []

        def urlopen(url, timeout):
            calls.append(url)
            return self.Response(b"approved")

        with (
            tempfile.TemporaryDirectory() as tmp,
            unittest.mock.patch.object(pilot.urllib.request, "urlopen", urlopen),
            unittest.mock.patch.object(pilot.time, "sleep", lambda s: None),
        ):
            part = Path(tmp) / "input.part"
            part.write_bytes(b"stale")
            pilot.download("https://example.invalid/x", part, 8)
            self.assertEqual(part.read_bytes(), b"approved")
            # One attempt: the stale file was removed up front, not after a failure.
            self.assertEqual(len(calls), 1)

    def test_a_truncated_body_is_retried(self):
        bodies = [b"appr", b"approved"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            unittest.mock.patch.object(
                pilot.urllib.request,
                "urlopen",
                lambda url, timeout: self.Response(bodies.pop(0)),
            ),
            unittest.mock.patch.object(pilot.time, "sleep", lambda s: None),
        ):
            part = Path(tmp) / "input.part"
            pilot.download("https://example.invalid/x", part, 8)
            self.assertEqual(part.read_bytes(), b"approved")
            self.assertEqual(bodies, [])

    def test_a_failed_attempt_is_retried_after_removing_the_partial_file(self):
        calls = []

        def urlopen(url, timeout):
            calls.append(url)
            if len(calls) == 1:
                raise OSError("Tunnel connection failed: 502 Bad Gateway")
            return self.Response(b"approved")

        with (
            tempfile.TemporaryDirectory() as tmp,
            unittest.mock.patch.object(pilot.urllib.request, "urlopen", urlopen),
            unittest.mock.patch.object(pilot.time, "sleep", lambda s: None),
        ):
            part = Path(tmp) / "input.part"
            part.write_bytes(b"stale")
            pilot.download("https://example.invalid/x", part, 8)
            self.assertEqual(part.read_bytes(), b"approved")
            self.assertEqual(len(calls), 2)

    def test_retries_are_bounded_and_oversized_bodies_refused(self):
        calls = []

        def urlopen(url, timeout):
            calls.append(url)
            return self.Response(b"x" * 9)

        with (
            tempfile.TemporaryDirectory() as tmp,
            unittest.mock.patch.object(pilot.urllib.request, "urlopen", urlopen),
            unittest.mock.patch.object(pilot.time, "sleep", lambda s: None),
        ):
            part = Path(tmp) / "input.part"
            with self.assertRaisesRegex(ValueError, "oversized"):
                pilot.download("https://example.invalid/x", part, 8, attempts=2)
            self.assertFalse(part.exists())
            self.assertEqual(len(calls), 2)

    def test_pins_name_no_mlx_source_archive(self):
        pins = json.loads((HERE / "pins.json").read_text())
        self.assertNotIn("mlx.tar.gz", pins["files"])
        self.assertFalse({"mlx_commit", "mlx_script", "mlx_script_sha256"} & set(pins))
        for spec in pins["files"].values():
            self.assertEqual(len(spec["sha256"]), 64)
            self.assertTrue(spec["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
