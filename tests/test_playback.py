"""Playback command timing and terminal diagnostic regression tests (CPU)."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

from humanoid.amp.playback import command_at, inject_command, speed_profile
from humanoid.amp.replay import ReplayRecorder


class PlaybackTests(unittest.TestCase):
    def test_six_stage_command_boundaries(self):
        profile = speed_profile(.02)
        self.assertEqual(sum(n for n, _ in profile), 3000)
        self.assertEqual(command_at(499, profile), (0, 0.))
        self.assertEqual(command_at(500, profile), (1, .4))
        self.assertEqual(command_at(1000, profile), (2, .6))
        self.assertEqual(command_at(2999, profile), (5, 0.))
        with self.assertRaises(ValueError):
            speed_profile(.02, 0.)

    def test_inject_only_latest_command_frame(self):
        history = deque([torch.ones(1, 96), torch.ones(1, 96)])
        env = SimpleNamespace(obs_history=history, obs_buf=torch.cat(list(history), -1),
                              commands=torch.zeros(1, 3), commands_scale=torch.tensor([2., 2., .25]),
                              command_ranges={}, cfg=SimpleNamespace(env=SimpleNamespace(num_single_obs=96)))
        before = env.obs_buf.clone()
        inject_command(env, .4, -.1, .2)
        torch.testing.assert_close(env.obs_buf[:, -96:-93], torch.tensor([[.8, -.2, .05]]))
        torch.testing.assert_close(env.obs_history[-1][:, :3], env.obs_buf[:, -96:-93])
        torch.testing.assert_close(env.obs_buf[:, :96], before[:, :96])
        torch.testing.assert_close(env.obs_buf[:, -93:], before[:, -93:])
        self.assertEqual(env.command_ranges["lin_vel_x"], [.4, .4])

    def test_terminal_csv_without_render(self):
        env = SimpleNamespace(
            dt=.02, dof_names=["joint"], feet_indices=torch.tensor([1, 0]), envs=[0], actor_handles=[0],
            gym=SimpleNamespace(get_actor_rigid_body_names=lambda *_: ["right_ankle", "left_ankle"]),
            commands=torch.tensor([[99., 99., 99.]]),
            cfg=SimpleNamespace(env=SimpleNamespace(), sim=SimpleNamespace(dt=.002),
                                normalization=SimpleNamespace(clip_actions=5.)))
        runner = SimpleNamespace(discriminator=SimpleNamespace(style_reward=lambda _: torch.tensor([.5])),
                                 disc_cfg={"style_reward_scale": 2., "task_reward_weight": .5})
        state = {"root": torch.tensor([[1., 2., .6, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]),
                 "velocity": torch.tensor([[.3, 0., 0.]]), "omega": torch.zeros(1, 3),
                 "gravity": torch.tensor([[0., 0., -1.]]), "q": torch.tensor([[.2]]),
                 "dq": torch.zeros(1, 1), "torque": torch.ones(1, 1), "action": torch.ones(1, 1),
                 "target": torch.tensor([[.4]]), "timeout": torch.tensor([True]),
                 "episode_step": torch.tensor([1001]), "torque_clip_step": torch.tensor([7]),
                 "command": torch.tensor([[.4, 0., 0.]]), "command_observed": torch.tensor([[.4, 0., 0.]]),
                 "contact": torch.tensor([[[0., 0., 20.], [0., 0., 30.]]]), "feet": torch.zeros(1, 2, 13)}
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ReplayRecorder(env, runner, tmp, render=False)
            with contextlib.redirect_stdout(io.StringIO()):
                recorder.write(0, torch.tensor([.04]), torch.tensor([True]),
                               {"replay_state": state, "amp_obs": torch.zeros(1)}, requested=.6, stage=2)
                recorder.close()
            with open(Path(tmp)/"isaac_diag.csv") as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["command_x"]), .4)
            self.assertEqual(float(row["command_requested"]), .6)
            self.assertEqual(float(row["foot_force_l"]), 20.)
            self.assertEqual(float(row["foot_force_r"]), 30.)
            self.assertAlmostEqual(float(row["pos_des_raw_joint"]), .4)
            self.assertEqual(int(row["clip_count"]), 7)
            metadata = json.loads((Path(tmp)/"replay_metadata.json").read_text())
            self.assertEqual(metadata["timeouts"], 1)
            self.assertEqual(metadata["captured_steps"], 1)
            self.assertFalse((Path(tmp)/"play_output.mp4").exists())
            with self.assertRaises(FileExistsError):
                ReplayRecorder(env, runner, tmp, render=False)
