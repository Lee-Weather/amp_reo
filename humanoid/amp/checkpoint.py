"""Prepare an authenticated runtime checkpoint without logging its URL."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def checkpoint_directory(args, default_directory):
    """Choose a fresh, fixed directory shared by input and output checkpoints."""
    requested = getattr(args, "checkpoint_dir", None)
    if not requested:
        return default_directory
    if default_directory is None:
        raise ValueError("checkpoint_dir requires a training log directory")
    if getattr(args, "resume", False):
        raise ValueError("Use a new fixed directory with warm_start; legacy resume uses its normal run layout")
    directory = Path(requested).resolve()
    source = getattr(args, "warm_start", None)
    initial = Path(source).resolve() if source else None
    if initial is not None and initial.parent != directory:
        raise ValueError("warm_start must be inside checkpoint_dir so SDK input/output monitoring shares one directory")
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError("checkpoint_dir is not a directory")
        allowed = {initial.name} if initial is not None else set()
        for extra_path, extra_digest in getattr(args, 'checkpoint_extra_inputs', {}).items():
            extra = Path(extra_path).resolve()
            if extra.parent != directory or not extra.is_file() or hashlib.sha256(extra.read_bytes()).hexdigest() != extra_digest:
                raise ValueError('Additional checkpoint input must be hash-verified in the persistent output directory')
            allowed.add(extra.name)
        if any(p.name not in allowed for p in directory.iterdir()):
            raise FileExistsError("checkpoint_dir already contains run artifacts; preserve it and choose a fresh directory")
    return str(directory)


def prepare_warm_start(args):
    # Validate before downloading or creating an environment, not after a
    # GPU has been allocated and old artifacts may have been overwritten.
    checkpoint_directory(args, "training")
    source = getattr(args, "warm_start", None)
    encoded = getattr(args, "warm_start_url_b64", None)
    expected = getattr(args, "warm_start_sha256", None)
    if encoded and not source:
        raise ValueError("A download fallback requires --warm_start")
    if not source:
        return
    destination = Path(source).resolve()
    if not destination.is_file():
        if not encoded or not expected:
            raise FileNotFoundError("Warm-start checkpoint missing; provide a mounted file or verified download fallback")
        try:
            url = base64.urlsafe_b64decode(encoded.encode()).decode()
            if not url.startswith("https://"):
                raise ValueError("Checkpoint download requires HTTPS")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="checkpoint_", dir=str(destination.parent)) as temporary:
                # The SDK locks its first discovered PT directory. Never
                # expose a temporary .pt in a directory that will be removed.
                staged = Path(temporary)/"checkpoint.part"
                with urllib.request.urlopen(url, timeout=90) as response, staged.open("wb") as output:
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        output.write(block)
                if hashlib.sha256(staged.read_bytes()).hexdigest() != expected:
                    raise ValueError("Downloaded checkpoint SHA256 mismatch")
                staged.replace(destination)
        except Exception:
            # Exception strings from urllib may contain signed download URLs.
            raise RuntimeError("Warm-start download failed; URL withheld") from None
    if expected and hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise ValueError("Warm-start checkpoint SHA256 mismatch")
    if getattr(args, "checkpoint_dir", None):
        # SDK watchdog only registers directories after a PT event. A mounted
        # input predates the watcher, and an atomic rename is a move event.
        # Emit a modification in the shared, persistent directory explicitly.
        # This changes only the private input copy's mtime, never its bytes.
        destination.touch()
        print("CHECKPOINT_INPUT_READY "+json.dumps({"path": str(destination),
              "bytes": destination.stat().st_size, "sha256": expected}), flush=True)
