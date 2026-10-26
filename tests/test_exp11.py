"""CPU checks of control semantics, buffer boundaries and metadata compatibility."""
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

from humanoid.amp.control import bounded_pd, StandingCommands, sync_policy_commands
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import FakeEnv, library, runner_config
from test_amp import MOTIONS, URDF, JOINTS, KEYS
from humanoid.amp.motion import MotionLibrary


class Exp11Tests(unittest.TestCase):
    def test_reference_grounding_uses_collision_mesh_and_recomputes_velocity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = library()
            grounded = MotionLibrary(MOTIONS, URDF, JOINTS, KEYS, ground_clearance=.002)
        state = grounded.frames[0, :30]
        minimum = grounded.grounding.minimum_height(state[:, :3], state[:, 3:7], state[:, 13:42])
        torch.testing.assert_close(minimum, torch.full_like(minimum, .002), atol=1e-6, rtol=0)
        torch.testing.assert_close(state[:, 13:], old.frames[0, :30, 13:])
        torch.testing.assert_close(state[1:-1, 9], (state[2:, 2]-state[:-2, 2])*25, atol=1e-6, rtol=1e-5)
        self.assertIsNone(old.grounding)
        self.assertGreater((state[:, 2]-old.frames[0, :30, 2]).abs().max().item(), .02)

    def test_pd_clips_targets_before_computing_torques(self):
        raw = torch.tensor([[2., -2., .1]])
        limits = torch.tensor([.4, .3, .2])
        target, torque, cost = bounded_pd(raw, torch.zeros_like(raw), torch.zeros_like(raw),
                                          -limits, limits, torch.ones(3)*10, torch.ones(3), torch.ones(3)*100)
        torch.testing.assert_close(target, torch.tensor([[.4, -.3, .1]]))
        torch.testing.assert_close(torque, torch.tensor([[4., -3., 1.]]))
        torch.testing.assert_close(cost, torch.tensor([[1.6**2, 1.7**2, 0.]]))
        # Verify velocity damping and motor torque saturation too.
        _, torque, _ = bounded_pd(raw, torch.zeros_like(raw), torch.ones_like(raw),
                                 -limits, limits, torch.ones(3)*10, torch.ones(3)*2, torch.ones(3))
        torch.testing.assert_close(torque, torch.tensor([[1., -1., -1.]]))

    def test_standing_buffer_exact_length_and_reset_boundary(self):
        buffer = StandingCommands(2, "cpu", 3)
        buffer.requested[:] = torch.tensor([[.4, .1, .2], [.6, 0., 0.]])
        buffer.reset(torch.tensor([0, 1]), torch.tensor([True, False]))
        for _ in range(3):
            torch.testing.assert_close(buffer.applied()[0], torch.zeros(3))
            self.assertAlmostEqual(buffer.applied()[1, 0].item(), .6)
            buffer.advance(torch.tensor([False, False]))
        torch.testing.assert_close(buffer.applied(), buffer.requested)
        buffer.reset(torch.tensor([0]), torch.tensor([True]))
        buffer.advance(torch.tensor([True, False]))
        self.assertEqual(buffer.remaining[0].item(), 3)

    def test_command_sync_updates_critic_and_preserves_old_history(self):
        env = SimpleNamespace(commands=torch.tensor([[.4, 0., .2]]), commands_scale=torch.tensor([2., 2., .25]),
                              obs_history=[torch.ones(1, 96), torch.ones(1, 96)], obs_buf=torch.ones(1, 192),
                              privileged_obs_buf=torch.ones(1, 99), cfg=SimpleNamespace(env=SimpleNamespace(num_single_obs=96)))
        sync_policy_commands(env)
        torch.testing.assert_close(env.obs_buf[:, -96:-93], torch.tensor([[.8, 0., .05]]))
        torch.testing.assert_close(env.privileged_obs_buf[:, :3], env.obs_buf[:, -96:-93])
        torch.testing.assert_close(env.obs_buf[:, :96], torch.ones(1, 96))

    def test_new_control_metadata_rejects_old_checkpoint_and_preserves_baseline(self):
        lib = library()
        old = AMPOnPolicyRunner(FakeEnv(lib), runner_config())
        env = FakeEnv(lib)
        env.checkpoint_control_metadata = lambda: {"controller": "bounded_pd_test", "hard_limits": [-1., 1.]}
        new = AMPOnPolicyRunner(env, runner_config())
        self.assertEqual(old._metadata()["schema_version"], 1)
        self.assertEqual(new._metadata()["schema_version"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"model.pt")
            old.save(path)
            with self.assertRaises(ValueError):
                new.load(path)
            new.save(path)
            new.load(path)
            with self.assertRaises(ValueError):
                old.load(path)
