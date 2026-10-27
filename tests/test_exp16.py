"""Moving reward must reject speed cheating without changing quiet-mode AMP."""
import unittest
from types import SimpleNamespace

import torch

from humanoid.amp.gait_rewards import moving_speed_quality, style_multiplier
from test_exp3 import definitions


class Exp16Tests(unittest.TestCase):
    def test_style_gate_rejects_rest_wrong_direction_and_large_overspeed(self):
        cmd = torch.tensor([0., .1, .1, -.2, -.2, .4])
        vx = torch.tensor([0., 0., .5, .2, -.2, .4])
        stand = torch.tensor([1., 0., 0., 0., 0., 0.])
        gate = style_multiplier(cmd, vx, stand)*moving_speed_quality(cmd, vx)
        torch.testing.assert_close(gate[[0, 1, 3, 4, 5]], torch.tensor([.2, 0., 0., 1., 1.]))
        self.assertLess(float(gate[2]), .001)
        for sigma in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                moving_speed_quality(cmd, vx, sigma)

    def test_metrics_report_applied_command_events_without_hiding_failures(self):
        class Parent:
            def compute_reward(self):
                self.extras = {"amp_style_multiplier": torch.ones(4), "training_metrics": {}}
        ns = dict(torch=torch, moving_speed_quality=moving_speed_quality, X1AMPExp15Env=Parent)
        definitions("humanoid/envs/x1/x1_amp_exp16_env.py", ns)
        env = ns["X1AMPExp16Env"]()
        env.cfg = SimpleNamespace(amp=SimpleNamespace(motion_style_velocity_sigma=.12))
        env.commands = torch.tensor([[-.2, 0., 0.], [.1, 0., 0.], [.4, 0., 0.], [0., 0., 0.]])
        env.base_lin_vel = torch.tensor([[-.8, 0., 0.], [0., 0., 0.], [.4, 0., 0.], [0., 0., 0.]])
        env.reset_buf = torch.tensor([1, 0, 1, 0])
        env.time_out_buf = torch.tensor([False, False, True, False])
        env.episode_length_buf = torch.tensor([50, 10, 1001, 20])
        env.compute_reward()
        m = env.extras["training_metrics"]
        torch.testing.assert_close(m["Episode/backward_non_timeout_fraction"], torch.tensor([1.]))
        torch.testing.assert_close(m["Episode/forward_non_timeout_fraction"], torch.tensor([0.]))
        self.assertEqual(m["Episode/slow_forward_non_timeout_fraction"].numel(), 0)
        torch.testing.assert_close(m["Tracking/backward_abs_vx_error"], torch.tensor([.6]))
        torch.testing.assert_close(env.extras["amp_style_multiplier"][-1], torch.tensor(1.))
        self.assertLess(float(env.extras["amp_style_multiplier"][0]), .001)


if __name__ == "__main__":
    unittest.main()
