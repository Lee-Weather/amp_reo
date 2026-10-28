"""Scratch curriculum, exploration bounds and absolute-iteration resume checks."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.runner import AMPOnPolicyRunner
from test_exp110 import DHFakeEnv, dh_config
from test_amp import library
from test_exp3 import definitions, experiment_config


class ParentRewards:
    def _reward_stand_lin_vel(self):
        return torch.ones(3)
    _reward_stand_ang_vel = _reward_stand_lin_vel
    _reward_stand_dof_vel = _reward_stand_lin_vel
    _reward_stand_pose = _reward_stand_lin_vel
    _reward_stand_orientation = _reward_stand_lin_vel

    def compute_reward(self):
        self.extras = {"amp_style_multiplier": torch.tensor([.2, .1, 0.]),
                       "training_metrics": {}}


def configuration():
    ns = experiment_config()
    for n in (14, 15, 16, 17, 18, 19, 110, 111, 112):
        definitions("humanoid/envs/x1/x1_amp_exp%d_config.py" % n, ns)
    return ns


class Exp112Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def test_production_config_preserves_interface_and_balances_updates(self):
        ns = configuration()
        parent, current = ns["X1AMPExp111Cfg"], ns["X1AMPExp112Cfg"]
        for key in ("env", "control", "asset", "commands", "amp", "sim", "noise"):
            self.assertIs(getattr(parent, key), getattr(current, key))
        self.assertIs(parent.rewards.scales, current.rewards.scales)
        cfg = ns["X1AMPExp112CfgPPO"]
        self.assertEqual((cfg.runner.num_steps_per_env, cfg.algorithm.num_learning_epochs,
                          cfg.algorithm.num_mini_batches), (24, 2, 4))
        self.assertEqual((cfg.policy.init_noise_std, cfg.policy.max_noise_std), (.3, .5))
        self.assertEqual(cfg.algorithm.entropy_coef, .001)
        self.assertEqual(cfg.discriminator.num_updates, 4)
        self.assertEqual(cfg.runner.max_iterations, 6000)
        self.assertFalse(cfg.runner.resume)
        self.assertEqual(ns["X1AMPExp111CfgPPO"].discriminator.num_updates, 16)

    def test_curriculum_stages_actual_rewards_and_style_without_resetting(self):
        ns = configuration()
        ns.update(torch=torch, X1AMPExp18Env=ParentRewards)
        definitions("humanoid/envs/x1/x1_amp_exp112_env.py", ns)
        env = ns["X1AMPExp112Env"]()
        env.cfg = ns["X1AMPExp112Cfg"]()
        env.commands = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [-.2, 0., 0.]])
        # Playback without collect_rollout must apply the full requirements.
        self.assertEqual(env._curriculum_progress, 1.)
        for iteration, progress, weight in ((0, 0., .1), (200, 0., .1),
                                            (850, .5, .55), (1500, 1., 1.), (6000, 1., 1.)):
            env.set_training_iteration(iteration)
            for name in ("lin_vel", "ang_vel", "dof_vel", "pose", "orientation"):
                torch.testing.assert_close(getattr(env, "_reward_stand_"+name)(), torch.full((3,), weight))
            env.compute_reward()
            expected = (1-progress)+progress*torch.tensor([.2, .1, 0.])
            torch.testing.assert_close(env.extras["amp_style_multiplier"], expected)
            metrics = env.extras["training_metrics"]
            torch.testing.assert_close(metrics["AMP/standing_style_multiplier"], expected[:1])
            torch.testing.assert_close(metrics["AMP/moving_style_multiplier"], expected[1:])
            torch.testing.assert_close(metrics["Curriculum/progress"], torch.full((3,), progress))
        with self.assertRaises(ValueError):
            env.set_training_iteration(-1)

    def test_noise_bound_in_both_real_ppo_paths_and_reported_distribution(self):
        for adaptive in (False, True):
            cfg = dh_config()
            cfg["policy"].update(init_noise_std=.49, max_noise_std=.5)
            cfg["algorithm"].update(entropy_coef=100., learning_rate=.05, desired_kl=100.)
            if adaptive:
                cfg["algorithm"].update(kl_check="minibatch", min_learning_rate=.01,
                                         max_learning_rate=.05, schedule="adaptive")
            runner = AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
            batch = runner.collect_rollout()
            metrics = runner.update_policy(batch)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
            self.assertLessEqual(runner.actor_critic.log_std.max().item(), math.log(.5)+1e-6)
            self.assertLessEqual(runner.actor_critic.distribution(batch["obs"]).scale.max().item(), .5)
            self.assertGreaterEqual(runner.actor_critic.noise_std().max().item(), .499)

    def test_reject_inconsistent_noise_floor_and_cap(self):
        for cap in (.1, 0., float("nan"), float("inf")):
            cfg = dh_config(); cfg["policy"].update(init_noise_std=.3, max_noise_std=cap)
            with self.assertRaisesRegex(ValueError, "upper bound"):
                AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
        cfg = dh_config()
        cfg["policy"].update(init_noise_std=.3, max_noise_std=.5,
            motion_exploration=dict(joint_names=["left_hip_pitch_joint"], std_minimum=.6, speed_threshold=.05))
        with self.assertRaisesRegex(ValueError, "floor exceeds"):
            AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)

    def test_checkpoint_resume_uses_absolute_iteration_and_rejects_other_curriculum(self):
        cfg = dh_config(); cfg["policy"].update(init_noise_std=.3, max_noise_std=.5)
        env = DHFakeEnv(self.lib)
        received = []
        env.set_training_iteration = received.append
        env.checkpoint_training_metadata = lambda: {"start": 200, "end": 1500}
        runner = AMPOnPolicyRunner(env, cfg)
        runner.current_learning_iteration = 850
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"model.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                runner.save(path)
                restored = AMPOnPolicyRunner(env, cfg)
                restored.load(path)
            restored.collect_rollout()
            self.assertEqual(received, [850])
            self.assertEqual(restored._metadata()["schema_version"], 4)
            env.checkpoint_training_metadata = lambda: {"start": 0, "end": 100}
            with self.assertRaises(ValueError):
                restored.load(path)


if __name__ == "__main__":
    unittest.main()
