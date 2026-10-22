"""Download and verify the separate, versioned X1 model/motion resource bundle."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_assets(root, manifest_path=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path or Path(__file__).with_name("assets_manifest.json"))
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = manifest["files"]
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name.startswith("czy/"):
            raise ValueError("Unsafe asset path in manifest")
    if all((root/name).is_file() and sha256(root/name) == digest for name, digest in expected.items()):
        print("X1 AMP assets verified: {} files".format(len(expected)), flush=True)
        return
    if not manifest["url"].startswith("https://"):
        raise ValueError("Asset download requires HTTPS")
    with tempfile.TemporaryDirectory(prefix="x1_amp_assets_") as temporary:
        archive = Path(temporary)/"assets.tar.gz"
        request = urllib.request.Request(manifest["url"], headers={"User-Agent": "X1-AMP-training/1.0"})
        print("Downloading versioned X1 AMP assets", flush=True)
        with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(archive) != manifest["sha256"]:
            raise ValueError("X1 AMP resource bundle SHA256 mismatch")
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("Unexpected files in X1 AMP resource bundle")
            if any(not member.isfile() for member in members):
                raise ValueError("Resource bundle must contain regular files only")
            # Check all contents before writing any assets into the checkout.
            for member in members:
                with tar.extractfile(member) as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
                if digest != expected[member.name]:
                    raise ValueError("Resource file SHA256 mismatch: "+member.name)
            for member in members:
                destination = (root/member.name).resolve()
                if root not in destination.parents:
                    raise ValueError("Asset path escapes project checkout")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print("X1 AMP assets installed and verified: {} files".format(len(expected)), flush=True)
