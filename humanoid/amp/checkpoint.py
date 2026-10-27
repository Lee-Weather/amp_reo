"""Prepare an authenticated runtime checkpoint without logging its URL."""
import base64
import hashlib
from pathlib import Path
import tempfile
import urllib.request


def prepare_warm_start(args):
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
                staged = Path(temporary)/"model.pt"
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
