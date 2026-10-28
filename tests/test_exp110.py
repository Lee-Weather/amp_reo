"""DH topology, supervised PPO/AMP updates, checkpoint and export regressions."""
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.distributions import Normal

from humanoid.amp.dh_models import ActorCriticDHAMP
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import FakeEnv, library, runner_config
from test_exp3 import definitions, experiment_config


def dh_config():
    cfg = runner_config()
    cfg["runner"]["policy_class_name"] = "ActorCriticDHAMP"
    cfg["policy"].update(in_channels=66, state_estimator_hidden_dims=[32],
                         kernel_size=[6, 4], filter_size=[32, 16], stride_size=[3, 2], lh_output_dim=64)
    cfg["algorithm"].update(state_estimator_loss_coef=1., state_estimator_obs_index=96)
    return cfg


class DHFakeEnv(FakeEnv):
    def __init__(self, lib):
        super().__init__(lib)
        self.num_obs, self.num_privileged_obs = 6336, 99
        self.obs, self.critic = torch.zeros(4, 6336), torch.zeros(4, 99)
        self.cfg.env = SimpleNamespace(frame_stack=66, short_frame_stack=5, num_single_obs=96)
        self.obs_scales = SimpleNamespace(lin_vel=2.)


class Exp110Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def test_config_preserves_control_rewards_and_amp(self):
        ns = experiment_config()
        for n in (14, 15, 16, 17, 18, 19, 110):
            definitions("humanoid/envs/x1/x1_amp_exp%d_config.py" % n, ns)
        parent, child = ns["X1AMPExp19Cfg"], ns["X1AMPExp110Cfg"]
        for key in ("control", "rewards", "asset", "commands", "amp", "sim", "domain_rand", "noise"):
            self.assertIs(getattr(parent, key), getattr(child, key))
        self.assertEqual(parent.env.num_observations, 480)
        self.assertEqual((child.env.frame_stack, child.env.short_frame_stack), (66, 5))
        self.assertEqual((child.env.num_observations, child.env.num_privileged_obs), (6336, 99))
        cfg = ns["X1AMPExp110CfgPPO"]
        self.assertEqual(cfg.policy.critic_hidden_dims, [768, 256, 128])
        self.assertEqual(cfg.runner.policy_class_name, "ActorCriticDHAMP")
        self.assertFalse(cfg.runner.resume)
        self.assertEqual((cfg.runner.num_steps_per_env, cfg.algorithm.lam), (96, .98))
        self.assertEqual(cfg.algorithm.state_estimator_obs_index, 96)
        self.assertEqual((cfg.runner.max_iterations, cfg.runner.save_interval), (6000, 100))

    def test_matches_original_dh_inference_and_uses_long_history(self):
        ns = dict(torch=torch, nn=nn, Normal=Normal)
        definitions("humanoid/algo/ppo/actor_critic_dh.py", ns)
        settings = dict(actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[768, 256, 128],
                        state_estimator_hidden_dims=[256, 128, 64], init_noise_std=.5,
                        in_channels=66, kernel_size=[6, 4], filter_size=[32, 16],
                        stride_size=[3, 2], lh_output_dim=64)
        current = ActorCriticDHAMP(6336, 99, 29, 480, 96, **settings)
        with contextlib.redirect_stdout(io.StringIO()):
            original = ns["ActorCriticDH"](480, 96, 99, 29, **settings)
        for name in ("actor", "critic", "long_history", "state_estimator"):
            getattr(original, name).load_state_dict(getattr(current, name).state_dict(), strict=True)
        obs = torch.randn(4, 6336)
        torch.testing.assert_close(current.act_inference(obs), original.act_inference(obs))
        self.assertEqual(current.actor[0].in_features, 547)
        self.assertEqual(current.long_history[0].in_channels, 66)
        changed = obs.clone(); changed[:, :96*10] += 10
        torch.testing.assert_close(current.estimate_velocity(obs), current.estimate_velocity(changed))
        self.assertFalse(torch.allclose(current.act_inference(obs), current.act_inference(changed)))

    def test_velocity_target_uses_scaled_critic_tail_without_privileged_leak(self):
        runner = AMPOnPolicyRunner(DHFakeEnv(self.lib), dh_config())
        for p in runner.actor_critic.state_estimator.parameters():
            nn.init.zeros_(p)
        obs = torch.randn(4, 6336)
        critic = torch.full((4, 99), 100.)
        critic[:, 96:99] = torch.tensor([1., 2., 3.])
        critic.requires_grad_(True)
        loss = runner._state_estimator_loss(obs, critic)
        self.assertAlmostEqual(loss.item(), 14/3, places=5)
        loss.backward()
        self.assertIsNone(critic.grad)
        self.assertGreater(runner.actor_critic.state_estimator[-1].bias.grad.abs().sum().item(), 0)
        self.assertEqual(runner.get_inference_policy()(obs).shape, (4, 29))

    def test_real_ppo_amp_updates_both_dh_branches_and_restores(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                cfg = dh_config()
                if adaptive:
                    cfg["algorithm"].update(kl_check="minibatch", schedule="adaptive",
                                             min_learning_rate=1e-5, max_learning_rate=3e-4)
                runner = AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
                before = copy.deepcopy(runner.actor_critic.state_dict())
                batch = runner.collect_rollout()
                self.assertEqual(batch["obs"].shape, (16, 6336))
                metrics = runner.update_policy(batch)
                self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
                self.assertGreater(metrics["state_estimator_loss"], 0.)
                for prefix in ("actor.", "critic.", "long_history.", "state_estimator."):
                    self.assertTrue(any(not torch.equal(v, before[k]) for k, v in runner.actor_critic.state_dict().items()
                                        if k.startswith(prefix)), prefix)
                self.assertTrue(all(math.isfinite(v) for v in runner.update_discriminator(batch["amp"]).values()))
                with tempfile.TemporaryDirectory() as tmp:
                    path = str(Path(tmp)/"model.pt")
                    runner.current_learning_iteration = 2
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.save(path)
                        restored = AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
                        restored.load(path)
                    self.assertEqual(restored.current_learning_iteration, 2)
                    self.assertEqual(restored._metadata()["policy"]["architecture"], "ActorCriticDHAMP")
                    for name, p in runner.actor_critic.named_parameters():
                        q = dict(restored.actor_critic.named_parameters())[name]
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        for key in ("step", "exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(runner.optimizer.state[p][key], restored.optimizer.state[q][key], rtol=0, atol=0)

    def test_mlp_checkpoint_rejected_before_mutating_dh(self):
        old = AMPOnPolicyRunner(FakeEnv(self.lib), runner_config())
        new = AMPOnPolicyRunner(DHFakeEnv(self.lib), dh_config())
        before = copy.deepcopy(new.actor_critic.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"old.pt")
            with contextlib.redirect_stdout(io.StringIO()): old.save(path)
            with self.assertRaisesRegex(ValueError, "MLP and DH"):
                new.warm_start(path)
        for key, value in new.actor_critic.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_complete_export_and_latest_command_exploration(self):
        cfg = dh_config()
        cfg["policy"]["init_noise_std"] = .01
        cfg["policy"]["motion_exploration"] = dict(joint_names=["left_hip_pitch_joint"], std_minimum=.25, speed_threshold=.05)
        runner = AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
        model = runner.actor_critic
        obs = torch.zeros(2, 6336); obs[0, 6240] = .8; obs[1, 0] = .8
        dist = model.distribution(obs)
        index = runner.env.dof_names.index("left_hip_pitch_joint")
        torch.testing.assert_close(dist.scale[:, index], torch.tensor([.25, .01]))
        torch.testing.assert_close(dist.mean, model.act_inference(obs))
        exported = torch.jit.script(copy.deepcopy(model.get_policy_module()).cpu().eval())
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"policy.jit"); exported.save(path); loaded = torch.jit.load(path)
            for n in (1, 3):
                x = torch.randn(n, 6336)
                torch.testing.assert_close(loaded(x), model.act_inference(x))

    def test_reject_invalid_supervision_and_frame_settings(self):
        for index in (None, 0, 95, 97):
            cfg = dh_config(); cfg["algorithm"]["state_estimator_obs_index"] = index
            with self.assertRaisesRegex(ValueError, "supervision"):
                AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
        for coef in (0., -1., float("inf"), float("nan")):
            cfg = dh_config(); cfg["algorithm"]["state_estimator_loss_coef"] = coef
            with self.assertRaisesRegex(ValueError, "loss_coef"):
                AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)
        cfg = dh_config(); cfg["policy"]["in_channels"] = 5
        with self.assertRaisesRegex(ValueError, "channels"):
            AMPOnPolicyRunner(DHFakeEnv(self.lib), cfg)

    def test_dh_log_keeps_episode_statistics_and_limits_duplicate_history_curves(self):
        runner = AMPOnPolicyRunner(DHFakeEnv(self.lib), dh_config())
        scalars = {}
        runner.writer = SimpleNamespace(add_scalar=lambda key, value, step: scalars.update({key: float(value)}), flush=lambda: None)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.learn(1)
        self.assertIn("Mean reward:", output.getvalue())
        self.assertIn("Mean episode length:", output.getvalue())
        self.assertIn("AMP state_estimator_loss:", output.getvalue())
        self.assertIn("Loss/state_estimator", scalars)
        feature_curves = [k for k in scalars if k.startswith("Observation/")]
        self.assertEqual(len(feature_curves), 192)
        self.assertIn("Observation/current_frame_mean_95", scalars)

    def test_onnx_source_loader_ignores_export_metadata_sidecar(self):
        ns = dict(os=os)
        definitions("humanoid/scripts/export_onnx_dh.py", ns)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/"run"; run.mkdir()
            (run/"policy.jit").touch(); (run/"policy.jit.json").touch()
            self.assertEqual(ns["get_load_path"](tmp), str(run/"policy.jit"))


if __name__ == "__main__":
    unittest.main()
