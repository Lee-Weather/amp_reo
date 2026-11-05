"""Phase-free guidance must reward demonstrated transitions, not held poses."""
import unittest

import torch
from humanoid.amp.reference_distance import ReferenceTransition
from humanoid.amp.stationary import CommandMotionSampler
from test_amp import library


class ReferenceDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()
        sampler = CommandMotionSampler(cls.lib, exclude_quiet_for_moving=True, relative_speed_tolerance=.5)
        cls.guide = ReferenceTransition(sampler)

    def test_real_supported_transitions_receive_dense_positive_credit(self):
        speed = self.guide.local_vx
        mask = (speed.abs() > .05) & (speed >= -.2) & (speed <= .65)
        histories = self.guide.histories[mask]
        bonus, score, baseline = self.guide.reward(histories, speed[mask])
        self.assertGreater(float(bonus.mean()), .02)
        self.assertGreater(float(score.mean()), .999)
        self.assertTrue((bonus >= -1e-6).all())
        self.assertTrue(torch.isfinite(baseline).all())

    def test_held_legs_and_sliding_have_zero_bonus_across_entire_command_range(self):
        commands = torch.linspace(-.2, .65, 1024)
        history = self.guide.histories[:1, -1:].expand(1024, self.lib.history_steps, -1).clone()
        history[:, :, self.guide.velocity_columns] = 0
        history[:, :, 4] = 1.5  # Body sliding cannot impersonate leg transitions.
        bonus, _, _ = self.guide.reward(history, commands)
        torch.testing.assert_close(bonus, torch.zeros_like(bonus), rtol=0, atol=1e-6)

    def test_zero_commands_never_receive_motion_guidance(self):
        history = self.guide.histories[:20]
        bonus, _, _ = self.guide.reward(history, torch.zeros(20))
        torch.testing.assert_close(bonus, torch.zeros(20), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "covers"):
            self.guide.reward(history, torch.full((20,), 10.))


if __name__ == "__main__":
    unittest.main()
