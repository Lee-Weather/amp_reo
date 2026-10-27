"""Check low-speed incentives, collision geometry and pre-reset AMP gating."""
import unittest
import torch

from humanoid.amp.gait_rewards import tracking_denominator, sole_heights, scuff_cost, style_multiplier
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import FakeEnv, library, runner_config
from test_exp3 import definitions, experiment_config


class Exp15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_small_speed_commands_no_longer_reward_rest_near_full_score(self):
        vx = torch.tensor([0., .1, -.2, .65])
        sigma = tracking_denominator(vx)
        reward = torch.exp(-vx.square()/sigma)
        self.assertEqual(float(reward[0]), 1.)
        self.assertLess(float(reward[1]), .5)
        self.assertLess(float(reward[2]), .3)
        torch.testing.assert_close(sigma[-1], torch.tensor(.09))
        torch.testing.assert_close(tracking_denominator(-vx), sigma)

    def test_collision_sole_height_includes_rotation_and_ground_offset(self):
        mesh = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.05, .05) for z in (-.1, .1)])
        feet = torch.zeros(2, 2, 13)
        feet[:, :, 2] = .1
        feet[:, :, 6] = 1
        torch.testing.assert_close(sole_heights(feet, [mesh, mesh], torch.zeros(2)), torch.zeros(2, 2))
        # A quarter turn about y brings the longer x extent into world z.
        feet[0, 0, 3:7] = torch.tensor([0., 2**-.5, 0., 2**-.5])
        heights = sole_heights(feet, [mesh, mesh], torch.zeros(2))
        self.assertAlmostEqual(float(heights[0, 0]), -.1, places=6)
        feet[:, :, 2] += .3
        torch.testing.assert_close(sole_heights(feet, [mesh, mesh], torch.full((2,), .3)), heights)

    def test_moving_style_requires_signed_progress_and_never_rewards_overspeed(self):
        commands = torch.tensor([0., .4, -.2, .1, .4, -.2, .01, .4])
        actual = torch.tensor([0., 0., 0., .05, .6, -.2, 0., -.2])
        gate = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        torch.testing.assert_close(style_multiplier(commands, actual, gate),
                                   torch.tensor([.2, 0., 0., .5, 1., 1., 1., 0.]))

    def test_scuff_cost_distinguishes_swing_support_and_zero_command(self):
        height = torch.tensor([[0., 0.], [.02, 0.], [0., 0.], [0., 0.]])
        speed = torch.tensor([[.4, 0.], [.4, 0.], [0., 0.], [.4, 0.]])
        vx = torch.tensor([.4, .4, .4, 0.])
        cost = scuff_cost(height, speed, vx)
        self.assertGreater(float(cost[0]), 0)
        torch.testing.assert_close(cost[1:], torch.zeros(3))
        torch.testing.assert_close(scuff_cost(height, speed, -vx), cost)
        # A terminal foot below the plane must not create an unbounded cost.
        torch.testing.assert_close(scuff_cost(torch.full_like(height, -.5), speed, vx)[0], cost[0])

    def test_leg_stationarity_is_not_diluted_by_arm_joints_or_applied_while_walking(self):
        ns = experiment_config()
        definitions("humanoid/envs/x1/x1_amp_exp14_config.py", ns)
        definitions("humanoid/envs/x1/x1_amp_exp15_config.py", ns)
        ns.update(torch=torch, X1AMPExp14Env=object)
        definitions("humanoid/envs/x1/x1_amp_exp15_env.py", ns)
        env = ns["X1AMPExp15Env"]()
        env.cfg = ns["X1AMPExp15Cfg"]()
        env.leg_indices = list(range(17, 29))
        env.stand_gate = torch.tensor([1., 0.])
        env.standing_state = torch.zeros(71)
        env.dof_pos = torch.zeros(2, 29)
        env.dof_vel = torch.zeros(2, 29)
        env.dof_vel[:, 17:] = .5
        torch.testing.assert_close(env._reward_stand_dof_vel(), torch.tensor([1., 0.]))
        env.dof_pos[:, 17:] = .2
        torch.testing.assert_close(env._reward_stand_pose(), torch.tensor([1., 0.]))
        env.dof_vel[:, :17] = 10
        torch.testing.assert_close(env._reward_stand_dof_vel(), torch.tensor([1., 0.]))

    def test_runner_uses_transition_multiplier_including_auto_reset_rows(self):
        class GatedEnv(FakeEnv):
            def step(self, actions):
                obs, critic, task, done, info = super().step(actions)
                info["amp_style_multiplier"] = torch.tensor([0., .2, .5, 1.])
                return obs, critic, task, done, info
        env = GatedEnv(library())
        cfg = runner_config()
        runner = AMPOnPolicyRunner(env, cfg)
        batch = runner.collect_rollout()
        multipliers = torch.tensor([0., .2, .5, 1.]).repeat(cfg["runner"]["num_steps_per_env"])
        expected = batch["style_scores"]*cfg["discriminator"]["style_reward_scale"]*env.dt*multipliers
        torch.testing.assert_close(batch["style_rewards"], expected)
        mask = ~batch["dones"].bool()
        weight = cfg["discriminator"]["task_reward_weight"]
        torch.testing.assert_close(batch["rewards"][mask],
                                   weight*batch["task_rewards"][mask]+(1-weight)*expected[mask])


if __name__ == "__main__":
    unittest.main()
