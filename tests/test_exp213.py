"""CPU checks for exp2.13 heading observation and reward configuration."""
import unittest

import torch

from humanoid.amp.heading import (heading_error, heading_observation,
                                  clearance_shortfall, contact_share)

try:
    from humanoid.envs.x1.x1_amp_config import X1AMPFinalCfg, X1AMPFinalCfgPPO
    from humanoid.envs.x1.x1_amp_exp213_config import (
        X1AMPExp213Cfg, X1AMPExp213CfgPPO, SOURCE_SHA256)
    HAVE_ENV = True
except ModuleNotFoundError:  # isaacgym is only available on the training host
    HAVE_ENV = False
    X1AMPFinalCfg = X1AMPFinalCfgPPO = X1AMPExp213Cfg = X1AMPExp213CfgPPO = object
    SOURCE_SHA256 = ""


class HeadingHelperTests(unittest.TestCase):
    def test_heading_error_wraps_to_pi(self):
        yaw = torch.tensor([3.0, -3.0, 0.1])
        ref = torch.tensor([-3.0, 3.0, 0.0])
        error = heading_error(yaw, ref)
        torch.testing.assert_close(error, torch.tensor([6.0 - 2 * torch.pi,
                                                        -6.0 + 2 * torch.pi, 0.1]),
                                   atol=1e-6, rtol=0)

    def test_heading_observation_clips_and_scales(self):
        error = torch.tensor([-2.0, -0.2, 0.0, 2.0])
        value = heading_observation(error, clip=0.8, scale=2.0)
        torch.testing.assert_close(value, torch.tensor([-1.6, -0.4, 0.0, 1.6]))

    def test_clearance_shortfall_is_swing_weighted(self):
        height = torch.tensor([[0.0, 0.0], [0.004, 0.010]])
        phase = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        cost = clearance_shortfall(height, phase, 0.005)
        torch.testing.assert_close(cost, torch.tensor([1.0, 0.02]), atol=1e-6, rtol=0)
        zeros = clearance_shortfall(torch.ones(1, 2) * 0.02, torch.zeros(1, 2), 0.005)
        torch.testing.assert_close(zeros, torch.zeros(1))

    def test_contact_share_handles_no_contact(self):
        share = contact_share(torch.tensor([[1.0, 3.0], [0.0, 0.0]]))
        torch.testing.assert_close(share, torch.tensor([[0.25, 0.75], [0.5, 0.5]]))


@unittest.skipUnless(HAVE_ENV, "isaacgym is not installed locally")
class Exp213ConfigTests(unittest.TestCase):
    def test_inherits_flat_final_config(self):
        self.assertTrue(issubclass(X1AMPExp213Cfg, X1AMPFinalCfg))
        self.assertTrue(issubclass(X1AMPExp213CfgPPO, X1AMPFinalCfgPPO))

    def test_observation_dimensions_unchanged(self):
        cfg = X1AMPExp213Cfg()
        self.assertEqual(cfg.env.num_single_obs, 98)
        self.assertEqual(cfg.env.num_observations, 66 * 98)
        self.assertEqual(cfg.env.num_privileged_obs, 101)
        self.assertEqual(cfg.env.frame_stack, 66)

    def test_training_changes(self):
        cfg = X1AMPExp213Cfg()
        self.assertEqual(cfg.amp.nominal_startup_fraction, 0.5)
        self.assertEqual(cfg.rewards.scales.corridor_yaw, -0.4)
        self.assertEqual(cfg.rewards.scales.corridor_lateral, -0.3)
        self.assertEqual(cfg.rewards.guide_weight_override, 0.3)
        self.assertEqual(cfg.rewards.raw_weight_override, 0.3)
        self.assertLess(cfg.rewards.heading_scale, 0)
        self.assertEqual(cfg.rewards.heading_obs_scale, 1.0)
        self.assertEqual(cfg.rewards.heading_clip, 2.0)
        self.assertGreater(cfg.rewards.heading_command_gain, 0)
        self.assertGreater(cfg.rewards.heading_command_clip, 0)
        self.assertLess(cfg.rewards.swing_clearance_penalty, 0)
        self.assertLess(cfg.rewards.contact_balance_scale, 0)

    def test_runner_and_source_contract(self):
        cfg = X1AMPExp213CfgPPO()
        self.assertEqual(cfg.runner_class_name, "AMPOnPolicyRunner")
        self.assertEqual(cfg.runner.warm_start_mode, "policy_only")
        self.assertEqual(cfg.runner.policy_transfer_source_iteration, 14500)
        self.assertEqual(cfg.runner.policy_transfer_source_task, "x1_amp_exp212")
        self.assertEqual(cfg.runner.policy_transfer_source_curriculum,
                         "nominal_startup_coverage_v1")
        self.assertEqual(cfg.runner.policy_transfer_source_sha256, SOURCE_SHA256)
        self.assertEqual(cfg.runner.max_iterations, 1000)
        self.assertEqual(cfg.algorithm.learning_rate, 3e-5)
        self.assertEqual(cfg.algorithm.max_learning_rate, 1e-4)
        self.assertEqual(cfg.algorithm.policy_transfer_warmup_iterations, 200)


if __name__ == "__main__":
    unittest.main()