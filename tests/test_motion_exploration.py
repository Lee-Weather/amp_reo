"""Keep Gaussian sampling, old checkpoints and KL consistent under gated noise."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch.distributions import kl_divergence

from humanoid.amp.models import ActorCritic
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import FakeEnv, library
from test_exp3 import adaptive_config


class MotionExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, spec=None):
        return ActorCritic(12, 15, 4, [8], [8], .1, motion_exploration=spec)

    def spec(self):
        return dict(indices=[1, 3], std_minimum=.25, command_index=0, command_threshold=.1)

    def test_old_weight_keys_and_deterministic_mean_are_unchanged(self):
        base, gated = self.model(), self.model(self.spec())
        self.assertEqual(set(base.state_dict()), set(gated.state_dict()))
        self.assertFalse(any("motion" in key for key in gated.state_dict()))
        gated.load_state_dict(base.state_dict(), strict=True)
        obs = torch.randn(16, 12)
        torch.testing.assert_close(gated.act_inference(obs), base.act_inference(obs), rtol=0, atol=0)
        critic_obs = torch.randn(16, 15)
        torch.testing.assert_close(gated.evaluate(critic_obs), base.evaluate(critic_obs), rtol=0, atol=0)
        torch.testing.assert_close(base.distribution(obs).scale, torch.full((16, 4), .1))

    def test_floor_only_affects_selected_joints_for_signed_moving_commands(self):
        model = self.model(self.spec())
        obs = torch.zeros(5, 12)
        obs[:, 0] = torch.tensor([0., .05, .1, -.2, .4])
        expected = torch.full((5, 4), .1)
        expected[3:, 1] = expected[3:, 3] = .25
        before = model.distribution(obs)
        torch.testing.assert_close(before.scale, expected)
        torch.testing.assert_close(before.loc, model.act_inference(obs))
        # A known mean-only shift must use the actual conditional variances.
        with torch.no_grad():
            model.actor[-1].bias.add_(.02)
        after = model.distribution(obs)
        torch.testing.assert_close(kl_divergence(before, after).sum(-1),
                                   (.02**2/(2*expected.square())).sum(-1), rtol=1e-4, atol=1e-6)

    def test_runner_maps_names_and_uses_identical_rollout_and_update_variances(self):
        env = FakeEnv(library())
        env.cfg.env = SimpleNamespace(num_single_obs=12)
        env.obs_scales = SimpleNamespace(lin_vel=2.)
        config = adaptive_config()
        config["policy"]["init_noise_std"] = .1
        names = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_pitch_joint", "right_knee_pitch_joint"]
        spec = dict(joint_names=names, std_minimum=.25, speed_threshold=.05)
        config["policy"]["motion_exploration"] = copy.deepcopy(spec)
        runner = AMPOnPolicyRunner(env, config)
        batch = runner.collect_rollout()
        self.assertEqual(config["policy"]["motion_exploration"], spec)
        moving = batch["obs"][:, 0].abs() > .1
        expected = torch.full_like(batch["old_std"], .1)
        for name in names:
            expected[moving, env.dof_names.index(name)] = .25
        torch.testing.assert_close(batch["old_std"], expected)
        self.assertIn("Policy/moving_selected_raw_target_noise_std_deg", runner.training_log.metric_sums)
        result = runner.update_policy(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def test_invalid_variance_and_joint_configuration_is_rejected(self):
        for changed in [dict(indices=[]), dict(indices=[1, 1]), dict(indices=[.5]),
                        dict(indices=[4]), dict(std_minimum=float("nan")), dict(std_minimum=0.),
                        dict(std_minimum=100.), dict(command_index=12), dict(command_threshold=-1.)]:
            spec = self.spec()
            spec.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.model(spec)


if __name__ == "__main__":
    unittest.main()
