"""CPU checks for PPO overshoot handling, resumption and reward conventions."""
import ast
import contextlib
import copy
import inspect
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import FakeEnv, ROOT, library, runner_config


def definitions(relative_path, namespace):
    """Load real definitions without importing Isaac Gym's env package on CPU."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), namespace)


def experiment_config():
    namespace = {"inspect": inspect}
    for path in ("humanoid/envs/base/base_config.py", "humanoid/envs/base/legged_robot_config.py",
                 "humanoid/envs/x1/x1_amp_config.py", "humanoid/envs/x1/x1_amp_exp11_config.py",
                 "humanoid/envs/x1/x1_amp_exp3_config.py"):
        definitions(path, namespace)
    return namespace


def adaptive_config():
    config = runner_config()
    namespace = experiment_config()
    algorithm = namespace["X1AMPExp3CfgPPO"]().algorithm
    for key in ("learning_rate", "schedule", "kl_check", "desired_kl", "min_learning_rate", "max_learning_rate"):
        config["algorithm"][key] = getattr(algorithm, key)
    config["discriminator"]["style_reward_scale"] = 10.
    return config


class Exp3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def test_x_only_commands_and_independent_speed_rewards(self):
        ns = experiment_config()
        cfg = ns["X1AMPExp3Cfg"]()
        baseline = ns["X1AMPCfg"]()
        self.assertEqual(baseline.commands.ranges.lin_vel_y, [-.1, .1])
        ns.update(torch=torch, X1AMPExp11Env=object)
        definitions("humanoid/envs/x1/x1_amp_exp3_env.py", ns)
        env = SimpleNamespace(cfg=cfg, commands=torch.empty(4096, 3), device="cpu")
        env.command_ranges = {key: getattr(cfg.commands.ranges, key)
                              for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
        # Execute the actual sampler and inherited angular reward on CPU.
        source = ast.parse((ROOT / "humanoid/envs/x1/x1_amp_env.py").read_text(encoding="utf-8"))
        body = next(node.body for node in source.body if isinstance(node, ast.ClassDef))
        methods = [node for node in body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_resample_commands", "_reward_tracking_ang_vel")]
        exec(compile(ast.Module(body=methods, type_ignores=[]), "<env methods>", "exec"), ns)
        ns["_resample_commands"](env, torch.arange(4096))
        self.assertTrue(((env.commands[:, 0] >= -.2) & (env.commands[:, 0] <= .65)).all())
        self.assertTrue((env.commands[:, 1:] == 0).all())
        self.assertTrue((env.commands[:, 0] < 0).any())
        self.assertTrue((env.commands[:, 0] > 0).any())
        env.commands[:] = torch.tensor([.6, 0., 0.])
        env.base_lin_vel = torch.tensor([[.44, 0., 0.]]).expand(4096, -1)
        env.base_ang_vel = torch.tensor([[0., 0., .16]]).expand(4096, -1)
        linear = ns["X1AMPExp3Env"]._reward_tracking_lin_vel(env)
        angular = ns["_reward_tracking_ang_vel"](env)
        self.assertAlmostEqual(linear[0].item(), math.exp(-.16**2/.09), places=6)
        self.assertAlmostEqual(angular[0].item(), math.exp(-.16**2/.25), places=6)
        self.assertEqual(cfg.amp.history_steps, 3)
        self.assertEqual(cfg.control.decimation, 10)
        self.assertEqual(cfg.commands.resampling_time, 6.)

    def test_high_kl_skips_all_updates_and_reduces_lr(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), adaptive_config())
        batch = runner.collect_rollout()
        saved = batch["old_mean"].clone()
        with torch.no_grad():
            runner.actor_critic.actor[-1].bias.add_(.5)
        before = copy.deepcopy(runner.actor_critic.state_dict())
        with patch.object(runner.optimizer, "step", wraps=runner.optimizer.step) as step:
            metrics = runner.update_policy(batch)
        self.assertEqual(step.call_count, 0)
        self.assertEqual(metrics["policy_updates"], 0)
        self.assertEqual(metrics["kl_early_stop"], 1.)
        self.assertAlmostEqual(metrics["kl"], 14.5, places=4)
        self.assertAlmostEqual(runner.optimizer.param_groups[0]["lr"], 1e-4/1.5)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        torch.testing.assert_close(batch["old_mean"], saved)
        for key, value in before.items():
            torch.testing.assert_close(runner.actor_critic.state_dict()[key], value)

    def test_overshoot_stops_next_minibatch_in_same_epoch(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), adaptive_config())
        batch = runner.collect_rollout()
        step = runner.optimizer.step
        def overshoot(*args, **kwargs):
            result = step(*args, **kwargs)
            with torch.no_grad():
                runner.actor_critic.actor[-1].bias.add_(.5)
            return result
        with patch.object(runner.optimizer, "step", side_effect=overshoot) as mocked:
            metrics = runner.update_policy(batch)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(metrics["policy_updates"], 1)
        self.assertEqual(metrics["kl_early_stop"], 1.)
        self.assertGreater(metrics["kl"], .03)

    def test_adaptive_lr_bounds_final_kl_and_resume(self):
        config = adaptive_config()
        config["algorithm"]["max_grad_norm"] = 0.  # isolate LR adaptation from parameter movement
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), config)
        batch = runner.collect_rollout()
        runner.optimizer.param_groups[0]["lr"] = 2e-5
        metrics = runner.update_policy(batch)
        self.assertEqual(metrics["policy_updates"], 4)
        self.assertAlmostEqual(metrics["kl"], 0., places=6)
        self.assertAlmostEqual(runner.optimizer.param_groups[0]["lr"], 3e-5)
        runner.optimizer.param_groups[0]["lr"] = 9e-5
        runner.update_policy(batch)
        self.assertEqual(runner.optimizer.param_groups[0]["lr"], 1e-4)
        runner.optimizer.param_groups[0]["lr"] = 1e-5
        with torch.no_grad():
            runner.actor_critic.actor[-1].bias.add_(.5)
        runner.update_policy(batch)
        self.assertEqual(runner.optimizer.param_groups[0]["lr"], 1e-5)
        runner.current_learning_iteration = 12
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp)/"model.pt")
            runner.save(path)
            restored = AMPOnPolicyRunner(FakeEnv(self.lib), config)
            restored.load(path)
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], 1e-5)
            self.assertEqual(restored.current_learning_iteration, 12)

    def test_style_score_independent_of_scaling_and_original_log_preserved(self):
        runner = AMPOnPolicyRunner(FakeEnv(self.lib), adaptive_config())
        runner.discriminator.style_reward = lambda amp: torch.full((len(amp),), .25)
        batch = runner.collect_rollout()
        torch.testing.assert_close(batch["style_scores"], torch.full((16,), .25))
        torch.testing.assert_close(batch["style_rewards"], torch.full((16,), .05))
        runner.disc_cfg["style_reward_scale"] = 2.
        batch = runner.collect_rollout()
        torch.testing.assert_close(batch["style_scores"], torch.full((16,), .25))
        torch.testing.assert_close(batch["style_rewards"], torch.full((16,), .01))
        values = {}
        runner.writer = SimpleNamespace(add_scalar=lambda key, value, step: values.update({key: float(value)}),
                                        flush=lambda: None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.learn(1)
        self.assertEqual(values["AMP/style_score"], .25)
        self.assertEqual(values["Train/mean_episode_length"], 4.)
        for key in ("AMP/kl_max", "AMP/kl_early_stop", "AMP/policy_updates", "Loss/learning_rate"):
            self.assertIn(key, values)
        self.assertIn("Mean reward:", output.getvalue())
        self.assertIn("Mean episode length:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
