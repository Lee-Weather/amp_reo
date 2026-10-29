"""Signed motion incentives and safe DH policy transfer across reward changes."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from humanoid.amp.gait_rewards import motion_velocity_reward
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import library
from test_exp110 import DHFakeEnv, dh_config
from test_exp112 import configuration, ParentRewards
from test_exp3 import definitions


def configuration113():
    ns = configuration()
    definitions("humanoid/envs/x1/x1_amp_exp113_config.py", ns)
    return ns


def transfer_config():
    cfg = dh_config()
    cfg["policy"].update(init_noise_std=.3, max_noise_std=.5)
    cfg["algorithm"].update(learning_rate=1e-5, min_learning_rate=1e-5,
                            max_learning_rate=1e-4, schedule="adaptive", kl_check="minibatch",
                            policy_transfer_warmup_iterations=200, policy_transfer_warmup_max_lr=1e-5)
    cfg["runner"].update(warm_start_mode="policy_only", require_policy_transfer=True,
                         policy_transfer_source_curriculum="source")
    return cfg


class Exp113Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def make_runner(self, transfer=True):
        cfg = transfer_config()
        if not transfer:
            cfg["runner"].clear()
            cfg["runner"].update(num_steps_per_env=4, save_interval=1, policy_class_name="ActorCriticDHAMP")
        env = DHFakeEnv(self.lib)
        env.checkpoint_training_metadata = lambda: {"curriculum": "target" if transfer else "source"}
        return AMPOnPolicyRunner(env, cfg)

    def save_source(self, directory):
        source = self.make_runner(False)
        with contextlib.redirect_stdout(io.StringIO()):
            source.learn(2)
            source.current_learning_iteration = 6000
            path = str(Path(directory)/"source.pt")
            source.save(path)
        return source, path

    def test_production_configuration_keeps_interface_and_sampling_budget(self):
        ns = configuration113()
        parent, cfg = ns["X1AMPExp112Cfg"], ns["X1AMPExp113Cfg"]
        for key in ("env", "control", "asset", "commands", "amp", "sim", "domain_rand", "noise"):
            self.assertIs(getattr(parent, key), getattr(cfg, key))
        self.assertEqual((cfg.env.num_observations, cfg.env.num_actions), (6336, 29))
        self.assertEqual((cfg.rewards.scales.backward_progress, cfg.rewards.scales.motion_progress), (0., 1.5))
        self.assertEqual(parent.rewards.scales.backward_progress, 1.5)
        ppo = ns["X1AMPExp113CfgPPO"]
        self.assertIs(ppo.policy, ns["X1AMPExp112CfgPPO"].policy)
        self.assertIs(ppo.discriminator, ns["X1AMPExp112CfgPPO"].discriminator)
        self.assertEqual((ppo.runner.num_steps_per_env, ppo.algorithm.num_learning_epochs,
                          ppo.algorithm.num_mini_batches, ppo.discriminator.num_updates), (24, 2, 4, 4))
        self.assertEqual((ppo.runner.max_iterations, ppo.runner.save_interval), (6000, 100))
        self.assertEqual(6000*4096*24, 589824000)
        self.assertTrue(ppo.runner.require_policy_transfer)

    def test_motion_credit_rest_target_direction_overspeed_and_zero(self):
        c = torch.cat((torch.linspace(-.2, -.05001, 100), torch.linspace(.05001, .65, 100)))
        rest = torch.zeros_like(c)
        torch.testing.assert_close(motion_velocity_reward(c, rest), rest)
        torch.testing.assert_close(motion_velocity_reward(c, c), torch.ones_like(c))
        self.assertTrue((motion_velocity_reward(c, .1*c) > 0).all())
        self.assertTrue((motion_velocity_reward(c, -c) < 0).all())
        self.assertTrue((motion_velocity_reward(c, 3*c) < 0).all())
        quiet = torch.tensor([-.05, 0., .05])
        torch.testing.assert_close(motion_velocity_reward(quiet, quiet+1), torch.zeros_like(quiet))
        for scale in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                motion_velocity_reward(c, c, scale)

    def test_zero_net_velocity_loops_do_not_earn_positive_mean_credit(self):
        c = torch.cat((torch.linspace(-.2, -.05001, 50, dtype=torch.float64),
                       torch.linspace(.05001, .65, 50, dtype=torch.float64)))[:, None]
        loop = torch.cat((c.expand(-1, 7), (-7*c/3).expand(-1, 3)), dim=1)
        self.assertLess(float(loop.mean(1).abs().max()), 1e-14)
        self.assertTrue((motion_velocity_reward(c, loop).mean(1) < 0).all())
        generator = torch.Generator().manual_seed(5)
        v = torch.randn(100, 1000, dtype=torch.float64, generator=generator)*.3
        v -= v.mean(1, keepdim=True)
        self.assertTrue((motion_velocity_reward(c, v).mean(1) <= 1e-12).all())

    def test_actual_environment_override_unlocks_amp_at_every_curriculum_stage(self):
        class Parent(ParentRewards):
            def compute_reward(self):
                self.extras = {"amp_style_multiplier": torch.zeros(4), "training_metrics": {
                    "AMP/moving_speed_quality": torch.zeros(2),
                    "Tracking/backward_progress_bonus": torch.ones(1),
                    "Tracking/backward_velocity_reward": torch.ones(1)}}
        ns = configuration113()
        ns.update(torch=torch, X1AMPExp18Env=Parent)
        definitions("humanoid/envs/x1/x1_amp_exp112_env.py", ns)
        ns["motion_velocity_reward"] = motion_velocity_reward
        definitions("humanoid/envs/x1/x1_amp_exp113_env.py", ns)
        env = ns["X1AMPExp113Env"]()
        env.cfg = ns["X1AMPExp113Cfg"]()
        env.commands = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [-.2, 0., 0.], [.03, 0., 0.]])
        env.stand_gate = torch.ones(4)
        env.base_lin_vel = torch.zeros(4, 3)
        env.root_states = torch.zeros(4, 13); env.root_states[:, 2] = .6
        env.env_origins = torch.zeros(4, 3)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]*4)
        env.reset_buf = torch.zeros(4, dtype=torch.long)
        env.time_out_buf = torch.zeros(4, dtype=torch.bool)
        for iteration in (0, 200, 850, 1500, 6000):
            env.set_training_iteration(iteration)
            env.compute_reward()
            torch.testing.assert_close(env.extras["amp_style_multiplier"], torch.tensor([.2, 1., 1., 1.]))
            self.assertNotIn("AMP/moving_speed_quality", env.extras["training_metrics"])
            self.assertNotIn("Tracking/backward_velocity_reward", env.extras["training_metrics"])
            torch.testing.assert_close(env._reward_motion_progress(), torch.zeros(4))
            env.base_lin_vel[:, 0] = env.commands[:, 0]
            torch.testing.assert_close(env._reward_motion_progress(), torch.tensor([0., 1., 1., 0.]))
            env.base_lin_vel.zero_()
        env.base_lin_vel[:, 0] = env.commands[:, 0]
        env.root_states[1, 2] = .2
        env.reset_buf[2] = 1
        torch.testing.assert_close(env._reward_motion_progress(), torch.zeros(4))
        env.time_out_buf[2] = True
        torch.testing.assert_close(env._reward_motion_progress(), torch.tensor([0., 0., 1., 0.]))

    def test_transfer_preserves_inference_but_resets_training_heads_and_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save_source(tmp)
            target = self.make_runner()
            critic = copy.deepcopy(target.actor_critic.critic.state_dict())
            disc = copy.deepcopy(target.discriminator.state_dict())
            obs = torch.randn(4, 6336)
            with contextlib.redirect_stdout(io.StringIO()):
                target.warm_start(path, hashlib.sha256(Path(path).read_bytes()).hexdigest())
            torch.testing.assert_close(target.actor_critic.act_inference(obs), source.actor_critic.act_inference(obs), rtol=0, atol=0)
            torch.testing.assert_close(target.actor_critic.log_std, source.actor_critic.log_std, rtol=0, atol=0)
            for key, value in critic.items():
                torch.testing.assert_close(target.actor_critic.critic.state_dict()[key], value, rtol=0, atol=0)
            for key, value in disc.items():
                torch.testing.assert_close(target.discriminator.state_dict()[key], value, rtol=0, atol=0)
            self.assertFalse(target.optimizer.state or target.disc_optimizer.state or target.replay.size)
            self.assertEqual(target.current_learning_iteration, 0)
            self.assertEqual(target.training_origin["source_iteration"], 6000)
            with self.assertRaisesRegex(ValueError, "fresh DH"):
                target.warm_start(path)
            # Exercise real PPO/AMP updates and exact resume of the new experiment.
            with contextlib.redirect_stdout(io.StringIO()):
                target.learn(1)
                new_path = str(Path(tmp)/"new.pt"); target.save(new_path)
                restored = self.make_runner(); restored.load(new_path)
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertEqual(restored.training_origin, target.training_origin)
            for key, value in target.actor_critic.state_dict().items():
                torch.testing.assert_close(restored.actor_critic.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(restored.optimizer.state and restored.disc_optimizer.state)
            self.assertTrue(all(torch.isfinite(p).all() for p in restored.actor_critic.parameters()))

    def test_incompatible_or_corrupt_source_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = self.save_source(tmp)
            source = torch.load(path, map_location="cpu")
            def corrupt(kind, state):
                if kind == "control": state["metadata"]["action_scale"] = .5
                if kind == "curriculum": state["metadata"]["training"]["curriculum"] = "other"
                if kind == "shape": state["model_state_dict"]["actor.0.weight"] = torch.zeros(1, 1)
                if kind == "nan": state["model_state_dict"]["log_std"][0] = float("nan")
            for kind in ("control", "curriculum", "shape", "nan", "hash", "pinned_hash"):
                with self.subTest(kind=kind):
                    state = copy.deepcopy(source); corrupt(kind, state)
                    bad_path = str(Path(tmp)/"bad.pt"); torch.save(state, bad_path)
                    target = self.make_runner()
                    if kind == "pinned_hash": target.cfg["policy_transfer_source_sha256"] = "0"*64
                    before = copy.deepcopy(target.actor_critic.state_dict())
                    with self.assertRaises(ValueError):
                        target.warm_start(bad_path, "0"*64 if kind == "hash" else None)
                    self.assertIsNone(target.training_origin)
                    for key, value in before.items():
                        torch.testing.assert_close(target.actor_critic.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "conventions"):
                self.make_runner().load(path)

    def test_warmup_caps_actual_adaptive_updates_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = self.save_source(tmp)
            target = self.make_runner()
            with contextlib.redirect_stdout(io.StringIO()): target.warm_start(path)
            for iteration, maximum in ((0, 1e-5), (199, 1e-5), (200, 1e-4)):
                target.current_learning_iteration = iteration
                target.optimizer.param_groups[0]["lr"] = maximum
                batch = target.collect_rollout()
                target.update_policy(batch)
                self.assertLessEqual(target.optimizer.param_groups[0]["lr"], maximum)
                self.assertEqual(target._policy_learning_rate_cap(), maximum)
            target.current_learning_iteration = 199
            with contextlib.redirect_stdout(io.StringIO()):
                saved = str(Path(tmp)/"resume.pt"); target.save(saved)
                restored = self.make_runner(); restored.load(saved)
            restored.update_policy(restored.collect_rollout())
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], 1e-5)

    def test_required_transfer_cannot_silently_train_from_scratch(self):
        runner = self.make_runner()
        with self.assertRaisesRegex(ValueError, "requires --warm_start"):
            runner.learn(1)
        self.assertEqual(runner.env.step_count, 0)


if __name__ == "__main__":
    unittest.main()
