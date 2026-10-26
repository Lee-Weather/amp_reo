"""CPU regression checks: python -m unittest discover -s tests -v."""
import copy
import contextlib
import io
import math
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.amp.motion import MotionLibrary, quat_inverse_rotate, quat_slerp
from humanoid.amp.models import ReplayBuffer, RunningNormalizer
from humanoid.amp.runner import AMPOnPolicyRunner

ROOT = Path(LEGGED_GYM_ROOT_DIR)
URDF = ROOT / "resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
MOTIONS = ROOT / "resources/motions/slow_walk_amp_v1.2/normalized"
JOINTS = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
KEYS = ["left_ankle_roll_link", "right_ankle_roll_link", "left_knee_pitch_link",
        "right_knee_pitch_link", "left_elbow_yaw_link", "right_elbow_yaw_link"]


def library(names=JOINTS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MotionLibrary(MOTIONS, URDF, names, KEYS)


class FakeEnv:
    """Auto-reset vector environment with distinct terminal and reset states."""
    def __init__(self, lib):
        self.motion_library = lib
        self.num_envs, self.num_obs, self.num_privileged_obs, self.num_actions = 4, 12, 15, 29
        self.dof_names, self.dt, self.device = JOINTS, 0.02, "cpu"
        self.default_dof_pos = torch.zeros(1, 29)
        self.cfg = SimpleNamespace(control=SimpleNamespace(action_scale=0.25))
        self.max_episode_length = 4
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.obs = torch.zeros(4, 12)
        self.critic = torch.zeros(4, 15)
        self.step_count = 0

    def reset(self):
        self.episode_length_buf.zero_()
        return self.obs, self.critic

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic

    def step(self, actions):
        self.step_count += 1
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        terminal = torch.ones_like(self.critic)
        self.obs.copy_(torch.randn_like(self.obs))
        self.critic.copy_(terminal)
        self.critic[done] = -100
        self.episode_length_buf[done] = 0
        return self.obs, self.critic, torch.full((4,), 0.02), done, {
            "amp_obs": self.motion_library.sample(4), "time_outs": done,
            "terminal_critic_obs": terminal}


def runner_config():
    return {"policy": {"actor_hidden_dims": [32], "critic_hidden_dims": [32], "init_noise_std": 0.5},
            "algorithm": {"learning_rate": 3e-4, "num_learning_epochs": 2, "num_mini_batches": 2,
                          "clip_param": 0.2, "gamma": 0.99, "lam": 0.95, "value_loss_coef": 1.,
                          "entropy_coef": 0.005, "max_grad_norm": 1., "desired_kl": 0.02},
            "discriminator": {"hidden_dims": [32], "learning_rate": 1e-4, "weight_decay": 1e-4,
                              "grad_penalty": 10., "batch_size": 8, "num_updates": 2,
                              "replay_size": 64, "style_reward_scale": 2., "task_reward_weight": 0.5},
            "runner": {"num_steps_per_env": 4, "save_interval": 1}}


class AMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def test_named_order_and_feature_conventions(self):
        ids = torch.tensor([0, 1, 2, 3])
        times = torch.full((4,), 0.5)
        states = self.lib.state(ids, times)
        reverse = library(list(reversed(JOINTS)))
        reversed_state = reverse.state(ids, times)
        torch.testing.assert_close(states[:, 13:42], reversed_state[:, 13:42].flip(-1))
        torch.testing.assert_close(self.lib.features(states)[:, -18:], reverse.features(reversed_state)[:, -18:])
        sample = self.lib.sample(128)
        self.assertEqual(sample.shape, (128, 3, 86))
        self.assertTrue(torch.isfinite(sample).all())
        torch.testing.assert_close(sample[:, :, 1:4].norm(dim=-1), torch.ones(128, 3))
        self.assertTrue((sample[:, :, 10:39] >= self.lib.fk.lower-1e-6).all())
        self.assertTrue((sample[:, :, 10:39] <= self.lib.fk.upper+1e-6).all())

    def test_no_clip_wrapping_and_quaternion_sign(self):
        ids = torch.tensor([0, 3])
        minimum = (self.lib.history_steps-1)*self.lib.dt
        history = self.lib.history(ids, torch.full((2,), minimum))
        first = self.lib.features(self.lib.state(ids, torch.zeros(2)))
        torch.testing.assert_close(history[:, 0], first)
        a = torch.tensor([[0., 0., math.sin(0.25), math.cos(0.25)]])
        torch.testing.assert_close(quat_slerp(a, -a, torch.tensor([[0.5]])), a)
        q = torch.tensor([[0., 0., math.sin(math.pi/4), math.cos(math.pi/4)]])
        torch.testing.assert_close(quat_inverse_rotate(q, torch.tensor([[1., 0., 0.]])),
                                   torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=1e-6)

    def test_reject_bad_joint_names(self):
        with self.assertRaises(ValueError):
            library(JOINTS[:-1])

    def test_normalization_and_replay_wrap(self):
        n = RunningNormalizer(2)
        x = torch.arange(20, dtype=torch.float).reshape(10, 2)
        n.update(x[:3]); n.update(x[3:])
        torch.testing.assert_close(n.mean, x.mean(0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(n.var, x.var(0, unbiased=False), atol=1e-3, rtol=1e-4)
        b = ReplayBuffer(5, (1,), "cpu")
        b.insert(torch.arange(4.).view(-1, 1)); b.insert(torch.arange(4., 8.).view(-1, 1))
        self.assertEqual(set(b.data.flatten().tolist()), {3., 4., 5., 6., 7.})

    def test_ppo_amp_update_and_checkpoint(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), runner_config())
        before = copy.deepcopy(runner.actor_critic.state_dict())
        batch = runner.collect_rollout()
        self.assertTrue((batch["dones"][-4:] == 1).all())
        metrics = runner.update_policy(batch)
        metrics.update(runner.update_discriminator(batch["amp"]))
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in runner.actor_critic.state_dict().items()))
        runner.current_learning_iteration = 7
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp)/"model.pt")
            runner.save(checkpoint)
            restored = AMPOnPolicyRunner(FakeEnv(self.lib), runner_config())
            restored.load(checkpoint)
            self.assertEqual(restored.current_learning_iteration, 7)
            for key, value in runner.discriminator.state_dict().items():
                torch.testing.assert_close(value, restored.discriminator.state_dict()[key])
            self.assertTrue(restored.optimizer.state_dict()["state"])
            restored.env.default_dof_pos += 0.1
            with self.assertRaises(ValueError):
                restored.load(checkpoint)

    def test_timeout_bootstraps_terminal_value(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), runner_config())
        # V(s)=first critic coordinate: terminal value 1, reset value -100.
        runner.actor_critic.evaluate = lambda x: x[:, 0]
        batch = runner.collect_rollout()
        expected = 0.5*batch["task_rewards"][-4:] + 0.5*batch["style_rewards"][-4:] + 0.99
        torch.testing.assert_close(batch["returns"][-4:], expected)

    def test_completed_episode_logging_across_rollouts(self):
        config = runner_config()
        config["runner"]["num_steps_per_env"] = 2
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), config)
        runner.actor_critic.evaluate = lambda x: x[:, 0]
        runner.discriminator.style_reward = lambda x: torch.zeros(len(x))
        # An old reset summary is retained by Isaac Gym on non-reset steps.
        original_step = runner.env.step
        def step(actions):
            obs, critic, reward, done, info = original_step(actions)
            info["episode"] = {"rew_tracking_lin_vel": torch.tensor(1.)}
            return obs, critic, reward, done, info
        runner.env.step = step
        runner.collect_rollout()
        self.assertFalse(runner.training_log.completed_lengths)
        self.assertFalse(runner.training_log.episode_infos)
        runner.collect_rollout()
        self.assertEqual(list(runner.training_log.completed_lengths), [4.] * 4)
        torch.testing.assert_close(torch.tensor(list(runner.training_log.returns[0])), torch.full((4,), 0.04))
        self.assertEqual(len(runner.training_log.episode_infos), 1)
        self.assertEqual(runner.training_log.lengths.sum().item(), 0.)

    def test_training_log_display_and_tensorboard(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), runner_config())
        values = {}
        runner.writer = SimpleNamespace(add_scalar=lambda key, value, step: values.update({key: float(value)}), flush=lambda: None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.learn(1)
        self.assertIn("Mean reward:", output.getvalue())
        self.assertIn("Mean episode length:", output.getvalue())
        self.assertIn("AMP disc_loss:", output.getvalue())
        self.assertIn("ETA:", output.getvalue())
        self.assertEqual(values["Train/mean_episode_length"], 4.)
        self.assertEqual(values["Perf/total_timesteps"], 16.)
        self.assertIn("Loss/value_function", values)
        self.assertIn("Observation/obs_mean_11", values)
        self.assertIn("AMP/task_reward", values)
        self.assertIn("Train/mean_reward/time", values)


if __name__ == "__main__":
    unittest.main()
