"""Cloud recovery must retain both optimizers and the original update target."""
import contextlib
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import torch
import test_exp113t as scratch_tests

spec = importlib.util.spec_from_file_location("resume_amp", Path(__file__).resolve().parents[1]/"humanoid/scripts/resume_amp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CloudRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch_tests.Exp113tTests.setUpClass()
        cls.fixture = scratch_tests.Exp113tTests()

    def test_saved_states_are_exact_then_both_optimizers_continue(self):
        source = self.fixture.make_runner()
        with contextlib.redirect_stdout(io.StringIO()):
            source.learn(2)
        source.current_learning_iteration = 5400
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            path = Path(td)/"model_5400.pt"; source.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = self.fixture.make_runner()
            proof = module.restore_continuation(target, path, digest, 5400, 6000, 600)
            self.assertEqual(proof["source_iteration"], 5400)
            self.assertIsNone(proof["training_origin"])
            for a, b in ((source.actor_critic, target.actor_critic), (source.discriminator, target.discriminator)):
                for key, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
            for a, b in ((source.optimizer, target.optimizer), (source.disc_optimizer, target.disc_optimizer)):
                original, restored = a.state_dict(), b.state_dict()
                self.assertEqual(original["param_groups"], restored["param_groups"])
                for key, value in original["state"].items():
                    for field, tensor in value.items():
                        torch.testing.assert_close(tensor, restored["state"][key][field], rtol=0, atol=0)
            target.learn(1)
            self.assertEqual(target.current_learning_iteration, 5401)
            self.assertGreater(min(float(v["step"]) for v in target.optimizer.state.values()), proof["optimizer_steps"]["policy"]["minimum"])
            self.assertGreater(min(float(v["step"]) for v in target.disc_optimizer.state.values()), proof["optimizer_steps"]["discriminator"]["minimum"])

    def test_wrong_budget_or_hash_does_not_modify_fresh_runner(self):
        target = self.fixture.make_runner()
        before = copy.deepcopy(target.actor_critic.state_dict())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"model.pt"; path.write_bytes(b"incorrect input")
            with self.assertRaisesRegex(ValueError, "original target"):
                module.restore_continuation(target, path, "0"*64, 5400, 6000, 599)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.restore_continuation(target, path, "0"*64, 5400, 6000, 600)
        for key, value in before.items():
            torch.testing.assert_close(value, target.actor_critic.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(target.optimizer.state)


if __name__ == "__main__":
    unittest.main()
