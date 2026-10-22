import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import shutil
import subprocess

from humanoid.amp.assets import ensure_assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.root = self.directory/"checkout"
        self.root.mkdir()
        self.content = b"test model data"
        self.name = "czy/robots/test.urdf"
        self.archive = self.directory/"assets.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            member = tarfile.TarInfo(self.name)
            member.size = len(self.content)
            tar.addfile(member, io.BytesIO(self.content))
        self.manifest = {"url": "https://example.com/assets.tar.gz",
                         "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                         "files": {self.name: hashlib.sha256(self.content).hexdigest()}}
        self.manifest_path = self.directory/"manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_download_verification_and_cached_assets(self):
        self.save_manifest()
        with patch("humanoid.amp.assets._download_once", side_effect=lambda url, dst: shutil.copyfile(self.archive, dst)):
            ensure_assets(self.root, self.manifest_path)
        self.assertEqual((self.root/self.name).read_bytes(), self.content)
        with patch("humanoid.amp.assets._download_once", side_effect=AssertionError("Must use valid cached assets")):
            ensure_assets(self.root, self.manifest_path)

    def test_corrupt_download_does_not_overwrite(self):
        target = self.root/self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing data")
        self.manifest["sha256"] = "0"*64
        self.save_manifest()
        with patch("humanoid.amp.assets._download_once", side_effect=lambda url, dst: shutil.copyfile(self.archive, dst)):
            with self.assertRaises(ValueError):
                ensure_assets(self.root, self.manifest_path)
        self.assertEqual(target.read_bytes(), b"existing data")

    def test_reject_unexpected_archive_members(self):
        self.manifest["files"] = {"czy/unexpected": hashlib.sha256(self.content).hexdigest()}
        self.save_manifest()
        with patch("humanoid.amp.assets._download_once", side_effect=lambda url, dst: shutil.copyfile(self.archive, dst)):
            with self.assertRaises(ValueError):
                ensure_assets(self.root, self.manifest_path)
        self.assertFalse((self.root/self.name).exists())

    def test_reject_path_traversal(self):
        self.manifest["files"] = {"czy/../../escape": "0"*64}
        self.save_manifest()
        with self.assertRaises(ValueError):
            ensure_assets(self.root, self.manifest_path)

    def test_endpoint_timeout_falls_back(self):
        self.manifest["mirrors"] = ["https://example.com/primary.tar.gz"]
        self.save_manifest()
        calls = []
        def download(url, destination):
            calls.append(url)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("download", 90)
            shutil.copyfile(self.archive, destination)
        with patch("humanoid.amp.assets._download_once", side_effect=download):
            ensure_assets(self.root, self.manifest_path)
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root/self.name).read_bytes(), self.content)


if __name__ == "__main__":
    unittest.main()
