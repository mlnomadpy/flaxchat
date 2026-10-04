"""Runtime compatibility metadata; no package download or model execution."""

import json
import unittest
from unittest.mock import patch
from scripts.build_runtime_bundle import host_platforms


class RuntimePlatformTests(unittest.TestCase):
    def test_complete_host_tags_preserve_intermediate_glibc_and_receipt(self):
        tags = [
            "cp312-cp312-manylinux_2_31_x86_64",
            "cp312-cp312-manylinux_2_28_x86_64",
            "cp312-cp312-manylinux_2_27_x86_64",
            "cp312-abi3-manylinux_2_27_x86_64",
            "cp312-cp312-manylinux2014_x86_64",
            "cp312-cp312-linux_x86_64",
            "py3-none-any",
        ]
        identity = dict(platform="linux", machine="x86_64", python="3.12.12")
        with patch(
            "scripts.build_runtime_bundle.subprocess.check_output",
            return_value=json.dumps(tags),
        ) as read:
            target = host_platforms(
                "/selected/python", identity, python_version="3.12", abi="cp312"
            )
        self.assertEqual(
            target["platforms"],
            [
                "manylinux_2_31_x86_64",
                "manylinux_2_28_x86_64",
                "manylinux_2_27_x86_64",
                "manylinux2014_x86_64",
                "linux_x86_64",
            ],
        )
        self.assertEqual(target["host_tags"], tags)
        self.assertIn("pip._vendor.packaging.tags", read.call_args.args[0][-1])
        self.assertEqual(target["abi"], "cp312")

    def test_auto_rejects_nonlinux_or_python_abi_mismatch_before_reads(self):
        with patch("scripts.build_runtime_bundle.subprocess.check_output") as read:
            for identity, kwargs in [
                (dict(platform="darwin", python="3.12.12"), {}),
                (dict(platform="linux", python="3.12.12"), dict(python_version="3.13")),
                (dict(platform="linux", python="3.12.12"), dict(abi="cp313")),
            ]:
                with self.assertRaises(ValueError):
                    host_platforms("/python", identity, **kwargs)
            read.assert_not_called()

    def test_official_python_artifact_requires_exact_origin_release_and_version(self):
        from scripts.prepare_python_artifact import validate_python_url

        url = "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/cpython-3.12.12+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"
        self.assertEqual(
            validate_python_url(url, "20260924", "3.12.12"), "python/bin/python3.12"
        )
        for bad in (
            url.replace("github.com", "example.com"),
            url.replace("20260924/", "latest/"),
            url.replace("3.12.12+", "3.13.0+"),
        ):
            with self.assertRaises(ValueError):
                validate_python_url(bad, "20260924", "3.12.12")

    def test_internal_interpreter_links_are_opt_in_and_escaping_links_rejected(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import io
        import tarfile
        from scripts.representation_run import extract_archive

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "python.tar"
            with tarfile.open(archive, "w") as handle:
                executable = tarfile.TarInfo("python/bin/python3.12")
                executable.size = 1
                handle.addfile(executable, io.BytesIO(b"x"))
                link = tarfile.TarInfo("python/bin/python3")
                link.type = tarfile.SYMTYPE
                link.linkname = "python3.12"
                handle.addfile(link)
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "strict")
            extract_archive(archive, root / "interpreter", allow_internal_links=True)
            self.assertEqual(
                (root / "interpreter/python/bin/python3").read_bytes(), b"x"
            )
            with tarfile.open(archive, "w") as handle:
                link.linkname = "../../../escape"
                handle.addfile(link)
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "bad", allow_internal_links=True)

    def test_headless_interpreter_preserves_runtime_and_keeps_safe_filters(self):
        import io
        import tarfile
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts.prepare_python_artifact import headless_archive
        from scripts.representation_run import digest, extract_archive

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, reduced = root / "original.tar", root / "headless.tar.gz"
            with tarfile.open(original, "w") as handle:
                for name, data in [
                    ("python/bin/python3.12", b"executable"),
                    ("python/lib/libpython.so", b"library"),
                    ("python/share/terminfo/a/adm1178", b"terminal"),
                ]:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    handle.addfile(entry, io.BytesIO(data))
                link = tarfile.TarInfo("python/share/terminfo/1/1178")
                link.type = tarfile.SYMTYPE
                link.linkname = "../a/adm1178"
                handle.addfile(link)
                link = tarfile.TarInfo("python/bin/python3")
                link.type = tarfile.SYMTYPE
                link.linkname = "python3.12"
                handle.addfile(link)
            receipt = headless_archive(original, reduced, digest(original))
            self.assertEqual(len(receipt["excluded_members"]), 2)
            self.assertFalse(receipt["interpreter_runtime_bytes_changed"])
            extract_archive(reduced, root / "unpacked", allow_internal_links=True)
            self.assertEqual(
                (root / "unpacked/python/bin/python3").read_bytes(), b"executable"
            )
            self.assertEqual(
                (root / "unpacked/python/lib/libpython.so").read_bytes(), b"library"
            )
            self.assertFalse((root / "unpacked/python/share/terminfo").exists())
            with self.assertRaises(ValueError):
                headless_archive(original, root / "tampered.tar.gz", "0" * 64)
            self.assertFalse((root / "tampered.tar.gz").exists())

    def test_headless_transform_does_not_allow_escape_in_excluded_terminal_files(self):
        import tarfile
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts.prepare_python_artifact import headless_archive
        from scripts.representation_run import digest

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "unsafe.tar"
            with tarfile.open(original, "w") as handle:
                link = tarfile.TarInfo("python/share/terminfo/escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../../../escape"
                handle.addfile(link)
            with self.assertRaisesRegex(ValueError, "Unsafe upstream interpreter link"):
                headless_archive(
                    original, root / "unsafe-headless.tar.gz", digest(original)
                )


if __name__ == "__main__":
    unittest.main()
