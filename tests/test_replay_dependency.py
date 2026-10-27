"""Never start queued playback on a live, failed or unverified training run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("replay_dependency", Path(__file__).resolve().parents[1]/"scripts/amp_tools/replay_dependency.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReplayDependencyTests(unittest.TestCase):
    def test_live_process_never_starts_playback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"train_recovery.log").write_text("SERVER_RECOVERY_OK\n")
            self.assertIsNone(module.prepare_checkpoint(root, True, 8000, root/"play/model.pt"))
            self.assertFalse((root/"play").exists())

    def test_stopped_training_requires_completion_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"train_recovery.log").write_text("Learning iteration 7999/8000\n")
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                module.prepare_checkpoint(root, False, 8000, root/"play/model.pt")

    def test_final_file_must_match_receipt_before_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/"model_8000.pt"
            source.write_bytes(b"known checkpoint")
            receipt = dict(iteration=8000, path=str(source), bytes=source.stat().st_size,
                           sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            (root/"train_recovery.log").write_text("CHECKPOINT_SAVED "+json.dumps(receipt)+"\nSERVER_RECOVERY_OK\n")
            destination = root/"play/model_8000.pt"
            result = module.prepare_checkpoint(root, False, 8000, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(result["sha256"], receipt["sha256"])
            source.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                module.prepare_checkpoint(root, False, 8000, destination)


if __name__ == "__main__":
    unittest.main()
