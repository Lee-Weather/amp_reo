"""Scratch versus transferred initialization with matched rewards and LR caps."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch

from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import library
from test_exp110 import DHFakeEnv
from test_exp113 import configuration113, transfer_config
from test_exp3 import definitions


def configuration113t():
    ns = configuration113()
    definitions("humanoid/envs/x1/x1_amp_exp113t_config.py", ns)
    return ns


class Exp113tTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def make_runner(self):
        cfg = transfer_config()
        cfg["algorithm"]["policy_transfer_warmup_for_scratch"] = True
        cfg["runner"].update(require_policy_transfer=False, require_fresh_initialization=True,
                             warm_start_mode="scratch", policy_transfer_source_curriculum=None,
                             policy_transfer_source_sha256=None)
        env = DHFakeEnv(self.lib)
        env.checkpoint_training_metadata = lambda: {"initialization": "scratch_exp113t_v1"}
        return AMPOnPolicyRunner(env, cfg)

    def test_scratch_control_keeps_all_reward_control_and_sampling_settings(self):
        ns = configuration113t()
        parent, current = ns["X1AMPExp113Cfg"], ns["X1AMPExp113tCfg"]
        for key in ("env", "control", "asset", "rewards", "commands", "amp", "sim", "noise", "domain_rand"):
            self.assertIs(getattr(parent, key), getattr(current, key))
        ppo, base = ns["X1AMPExp113tCfgPPO"], ns["X1AMPExp113CfgPPO"]
        self.assertIs(ppo.policy, base.policy)
        self.assertIs(ppo.discriminator, base.discriminator)
        for key in dir(base.algorithm):
            if not key.startswith("_"):
                self.assertEqual(getattr(ppo.algorithm, key), getattr(base.algorithm, key))
        self.assertTrue(ppo.algorithm.policy_transfer_warmup_for_scratch)
        self.assertFalse(ppo.runner.require_policy_transfer or ppo.runner.resume)
        self.assertTrue(ppo.runner.require_fresh_initialization)
        self.assertEqual(ppo.runner.warm_start_mode, "scratch")
        self.assertIsNone(ppo.runner.policy_transfer_source_sha256)
        self.assertEqual((ppo.runner.max_iterations, ppo.runner.num_steps_per_env,
                          ppo.algorithm.num_learning_epochs, ppo.algorithm.num_mini_batches), (6000, 24, 2, 4))
        # Only checkpoint identity differs, not environment behavior.
        class Parent:
            def checkpoint_training_metadata(self): return {"motion_reward": "normalized_huber_progress_v1"}
        ns["X1AMPExp113Env"] = Parent
        definitions("humanoid/envs/x1/x1_amp_exp113t_env.py", ns)
        self.assertEqual(ns["X1AMPExp113tEnv"]().checkpoint_training_metadata(), {
            "motion_reward": "normalized_huber_progress_v1", "initialization": "scratch_exp113t_v1"})

    def test_no_checkpoint_can_be_warm_started_and_scratch_training_is_real(self):
        runner = self.make_runner()
        self.assertIsNone(runner.training_origin)
        torch.testing.assert_close(runner.actor_critic.noise_std(), torch.full((29,), .3))
        before = copy.deepcopy(runner.actor_critic.actor.state_dict())
        with self.assertRaisesRegex(ValueError, "does not accept --warm_start"):
            runner.warm_start("does-not-exist.pt")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.learn(2)
        self.assertIn("FRESH_START_OK", output.getvalue())
        self.assertIn('"source_checkpoint": null', output.getvalue())
        self.assertIn("Mean reward:", output.getvalue())
        self.assertEqual(runner.current_learning_iteration, 2)
        self.assertTrue(runner.optimizer.state and runner.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in runner.actor_critic.actor.state_dict().items()))
        self.assertTrue(all(torch.isfinite(p).all() for p in runner.actor_critic.parameters()))
        self.assertEqual(runner.optimizer.param_groups[0]["lr"], 1e-5)

    def test_same_warmup_cap_without_transfer_and_exact_resume(self):
        runner = self.make_runner()
        for iteration, cap in ((0, 1e-5), (199, 1e-5), (200, 1e-4)):
            runner.current_learning_iteration = iteration
            self.assertEqual(runner._policy_learning_rate_cap(), cap)
            runner.optimizer.param_groups[0]["lr"] = cap
            runner.update_policy(runner.collect_rollout())
            self.assertLessEqual(runner.optimizer.param_groups[0]["lr"], cap)
        runner.current_learning_iteration = 199
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"scratch.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                runner.save(path)
                restored = self.make_runner(); restored.load(path)
            self.assertIsNone(restored.training_origin)
            self.assertEqual(restored.current_learning_iteration, 199)
            self.assertEqual(restored._policy_learning_rate_cap(), 1e-5)
            with contextlib.redirect_stdout(io.StringIO()): restored.learn(1)
            self.assertEqual(restored.current_learning_iteration, 200)
            self.assertEqual(restored._policy_learning_rate_cap(), 1e-4)

    def test_transferred_state_is_rejected_in_scratch_control(self):
        runner = self.make_runner()
        runner.training_origin = {"mode": "policy_transfer_fresh_critic_amp_optimizers"}
        with self.assertRaisesRegex(ValueError, "must not inherit"):
            runner.learn(1)
        self.assertEqual(runner.env.step_count, 0)
        runner = self.make_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"other.pt")
            source = self.make_runner()
            source.env.checkpoint_training_metadata = lambda: {"curriculum": "policy_transfer_standing_v1"}
            with contextlib.redirect_stdout(io.StringIO()): source.save(path)
            with self.assertRaisesRegex(ValueError, "conventions"):
                runner.load(path)


if __name__ == "__main__":
    unittest.main()
