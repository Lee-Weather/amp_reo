"""Reference guidance must close cycles and preserve derivative/time conventions."""
import unittest

import torch
from humanoid.amp.phase_reference import PhaseReference
from test_amp import library


class PhaseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()
        clip = cls.lib.names.index("F1_B1_stand_to_walk")
        cls.stand = cls.lib.state(torch.tensor([clip]), torch.tensor([11/50]))[0, 13:42]
        cls.guide = PhaseReference(cls.lib, cls.stand, [
            ("F1_B3_continuous_walk", 27, 90), ("F1_B4_walk_backwards", 235, 305)])

    def test_zero_command_and_legal_targets_across_speed_range(self):
        command = torch.linspace(-.2, .65, 4096)
        phase = torch.linspace(-2, 3, 4096)
        q, velocity = self.guide.sample(command, phase)
        self.assertEqual(q.shape, (4096, 29))
        self.assertTrue(torch.isfinite(q).all() and torch.isfinite(velocity).all())
        self.assertTrue((q >= self.lib.fk.lower).all() and (q <= self.lib.fk.upper).all())
        q0, v0 = self.guide.sample(torch.zeros(4), torch.rand(4))
        torch.testing.assert_close(q0, self.stand.expand(4, -1))
        torch.testing.assert_close(v0, torch.zeros_like(v0))
        torch.testing.assert_close(self.guide.phase_rate(torch.zeros(4)), torch.zeros(4))

    def test_wrap_is_continuous_and_velocity_uses_warped_seconds(self):
        command = torch.tensor([.1, .6, -.1, -.2])
        q0, v0 = self.guide.sample(command, torch.zeros(4))
        q1, v1 = self.guide.sample(command, torch.ones(4))
        torch.testing.assert_close(q0, q1, rtol=0, atol=0)
        torch.testing.assert_close(v0, v1, rtol=0, atol=0)
        epsilon = 1e-5
        before, before_v = self.guide.sample(command, torch.full((4,), 1-epsilon))
        after, after_v = self.guide.sample(command, torch.full((4,), epsilon))
        self.assertLess(float((before-after).abs().max()), .001)
        self.assertLess(float((before_v-after_v).abs().max()), .01)
        phase = torch.full((4,), .372)
        rate = self.guide.phase_rate(command)
        dt = 1e-3
        qa, _ = self.guide.sample(command, phase-rate*dt)
        qb, _ = self.guide.sample(command, phase+rate*dt)
        q, velocity = self.guide.sample(command, phase)
        away_from_limits = (q > self.lib.fk.lower+.01) & (q < self.lib.fk.upper-.01)
        torch.testing.assert_close(((qb-qa)/(2*dt))[away_from_limits], velocity[away_from_limits], rtol=.015, atol=.002)

    def test_pose_selection_uses_observed_motion_instead_of_an_arbitrary_clock(self):
        commands = torch.tensor([.6, -.2])
        phase = torch.tensor([7/64, 23/64])
        q, velocity = self.guide.sample(commands, phase)
        inferred = self.guide.nearest_phase(q, velocity, commands)
        torch.testing.assert_close(inferred, phase, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
