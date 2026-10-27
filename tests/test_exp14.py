"""Check standing gates, real reference matching and safe warm-start semantics."""
import base64
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch

from humanoid.amp.checkpoint import prepare_warm_start
from humanoid.amp.motion import MotionLibrary, quat_inverse_rotate
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.amp.stationary import CommandMotionSampler, sample_commands
from test_amp import FakeEnv, JOINTS, KEYS, MOTIONS, URDF, runner_config
from test_exp3 import adaptive_config, definitions, experiment_config


class Exp14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.lib = MotionLibrary(MOTIONS, URDF, JOINTS, KEYS, ground_clearance=.002)
        cls.sampler = CommandMotionSampler(cls.lib)

    def test_command_probability_and_fixed_evaluation_command(self):
        torch.manual_seed(5)
        ranges = {"lin_vel_x": [-.2, .65], "lin_vel_y": [0., 0.], "ang_vel_yaw": [0., 0.]}
        commands = sample_commands(ranges, 20000, "cpu", .2)
        self.assertLess(abs(float((commands[:, 0] == 0).float().mean())-.2), .015)
        self.assertTrue((commands[:, 1:] == 0).all())
        self.assertGreater(float(commands[:, 0].min()), -.2)
        self.assertLess(float(commands[:, 0].max()), .65)
        ranges["lin_vel_x"] = [.4, .4]
        torch.testing.assert_close(sample_commands(ranges, 1024, "cpu", 1.)[:, 0], torch.full((1024,), .4))

    def test_real_rsi_matches_speed_and_quiet_history(self):
        commands = torch.zeros(4096, 3)
        commands[:, 0] = torch.linspace(-.2, .65, 4096)
        commands[:512] = 0
        ids, times = self.sampler.sample_matching(commands)
        state = self.lib.state(ids, times)
        vx = quat_inverse_rotate(state[:, 3:7], state[:, 7:10])[:, 0]
        self.assertLessEqual(float((vx-commands[:, 0]).abs().max()), .150001)
        directional = commands[:, 0].abs() > .05
        self.assertTrue((commands[directional, 0]*vx[directional] > 0).all())
        history = self.lib.history(ids[:512], times[:512])
        self.assertEqual(history.shape, (512, 3, 86))
        self.assertLessEqual(float(history[..., 4:7].norm(dim=-1).max()), .050001)
        self.assertLessEqual(float(history[..., 7:10].norm(dim=-1).max()), .300001)
        self.assertLessEqual(float(history[..., 39:68].square().mean(-1).sqrt().max()), .200001)
        torch.testing.assert_close(history[:, -1], self.lib.features(state[:512]))
        self.assertTrue((state[:512, 42:].abs() > 0).any())  # Genuine quiet motion, not erased velocities.

    def test_expert_quiet_windows_and_runner_hook(self):
        sampler = CommandMotionSampler(self.lib, expert_standing_probability=1.)
        history = sampler.sample_expert(256)
        self.assertLessEqual(float(history[..., 4:7].norm(dim=-1).max()), .050001)
        self.assertLessEqual(float(history[..., 39:68].square().mean(-1).sqrt().max()), .200001)
        env = FakeEnv(self.lib)
        env.sample_amp_expert = sampler.sample_expert
        runner = AMPOnPolicyRunner(env, runner_config())
        with patch.object(env, "sample_amp_expert", wraps=env.sample_amp_expert) as hook:
            result = runner.update_discriminator(self.lib.sample(16))
        self.assertEqual(hook.call_count, 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def standing_env(self):
        namespace = experiment_config()
        definitions("humanoid/envs/x1/x1_amp_exp14_config.py", namespace)
        class Parent:
            def compute_reward(self):
                self.extras = {"training_metrics": {}}
        namespace.update(torch=torch, X1AMPExp3Env=Parent)
        definitions("humanoid/envs/x1/x1_amp_exp14_env.py", namespace)
        env = namespace["X1AMPExp14Env"]()
        env.cfg = namespace["X1AMPExp14Cfg"]()
        env.dt, env.device, env.num_envs = .02, "cpu", 3
        env.commands = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [.01, 0., 0.]])
        env.zero_steps = torch.zeros(3, dtype=torch.long)
        env.stop_hold_steps = torch.zeros_like(env.zero_steps)
        env.previous_moving = torch.zeros(3, dtype=torch.bool)
        env.stop_pending = torch.zeros(3, dtype=torch.bool)
        env.base_lin_vel = torch.zeros(3, 3)
        env.base_ang_vel = torch.zeros(3, 3)
        env.dof_pos = torch.zeros(3, 29)
        env.dof_vel = torch.zeros_like(env.dof_pos)
        env.standing_state = torch.zeros(71)
        env.standing_gravity = torch.tensor([0., 0., -1.])
        env.projected_gravity = env.standing_gravity.expand(3, -1).clone()
        env.contact_forces, env.feet_indices = torch.zeros(3, 2, 3), torch.tensor([0, 1])
        return env

    def test_standing_ramp_reward_gates_and_stopping_measurement(self):
        env = self.standing_env()
        env.compute_reward()
        torch.testing.assert_close(env.stand_gate, torch.tensor([.04, 0., 0.]))
        for _ in range(24):
            env.compute_reward()
        torch.testing.assert_close(env.stand_gate, torch.tensor([1., 0., 0.]))
        for name in ("lin_vel", "ang_vel", "dof_vel", "pose", "orientation"):
            torch.testing.assert_close(getattr(env, "_reward_stand_"+name)(), torch.zeros(3))
        env.dof_vel[:] = 1
        env.base_lin_vel[:] = .1
        env.dof_pos[:] = .2
        for name in ("lin_vel", "dof_vel", "pose"):
            value = getattr(env, "_reward_stand_"+name)()
            self.assertGreater(float(value[0]), 0)
            torch.testing.assert_close(value[1:], torch.zeros(2))
        env.commands[:] = 0
        env.base_lin_vel[:] = 0
        for _ in range(10):
            env.compute_reward()
        self.assertEqual(env.extras["training_metrics"]["Stop/braking_time_s"].numel(), 2)
        env.compute_reward()
        self.assertEqual(env.extras["training_metrics"]["Stop/braking_time_s"].numel(), 0)

    def test_warm_start_retains_weights_and_resets_both_optimizers(self):
        config = adaptive_config()
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), config)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.learn(1)
        runner.current_learning_iteration = 6000
        runner.optimizer.param_groups[0]["lr"] = 1e-5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"model_6000.pt"
            runner.save(path)
            config = copy.deepcopy(config)
            config["algorithm"]["learning_rate"] = 3e-5
            restored = AMPOnPolicyRunner(FakeEnv(self.lib), config)
            with contextlib.redirect_stdout(io.StringIO()):
                restored.warm_start(path, hashlib.sha256(path.read_bytes()).hexdigest())
            for a, b in ((runner.actor_critic, restored.actor_critic), (runner.discriminator, restored.discriminator)):
                for key, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(restored.current_learning_iteration, 6000)
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], 3e-5)
            self.assertFalse(restored.optimizer.state)
            self.assertFalse(restored.disc_optimizer.state)
            with contextlib.redirect_stdout(io.StringIO()):
                restored.learn(1)
            self.assertEqual(restored.current_learning_iteration, 6001)
            restored.save(path)
            again = AMPOnPolicyRunner(FakeEnv(self.lib), config)
            again.load(path)
            self.assertEqual(again.training_origin["source_iteration"], 6000)
            self.assertEqual(again.current_learning_iteration, 6001)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                restored.warm_start(path, "0"*64)

    def test_download_verifies_sha_and_does_not_disclose_signed_url(self):
        payload = b"checkpoint test payload"
        secret_url = "https://example.invalid/checkpoint?signature=PRIVATE"
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(warm_start=str(Path(temporary)/"model.pt"),
                                   warm_start_url_b64=base64.urlsafe_b64encode(secret_url.encode()).decode(),
                                   warm_start_sha256=hashlib.sha256(payload).hexdigest())
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
                prepare_warm_start(args)
            self.assertEqual(Path(args.warm_start).read_bytes(), payload)
            args.warm_start = str(Path(temporary)/"another.pt")
            with patch("urllib.request.urlopen", side_effect=RuntimeError(secret_url)):
                with self.assertRaises(RuntimeError) as failure:
                    prepare_warm_start(args)
            self.assertNotIn("PRIVATE", str(failure.exception))
            self.assertFalse(Path(args.warm_start).exists())


if __name__ == "__main__":
    unittest.main()
