"""Absolute phase credit, collision-sole support, phase RSI and transfer contract."""
import ast
import inspect
import math
import unittest
import warnings
from pathlib import Path

import torch

from humanoid.amp.motion import MotionLibrary, URDFKinematics
from humanoid.amp.control import StandingCommands
from humanoid.amp.phase_initialization import ForwardPhaseBank
from humanoid.amp.phase_reference import PhaseReference
from humanoid.amp.phase_rewards import (absolute_phase_credit, foot_support_score,
                                       static_cycle_baseline, static_velocity_baseline)
from test_amp import JOINTS, KEYS, MOTIONS, ROOT, URDF, library


def definitions(relative_path, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), namespace)


def configuration():
    namespace = {"inspect": inspect}
    paths = ["humanoid/envs/base/base_config.py", "humanoid/envs/base/legged_robot_config.py",
             "humanoid/envs/x1/x1_amp_config.py", "humanoid/envs/x1/x1_amp_exp11_config.py",
             "humanoid/envs/x1/x1_amp_exp3_config.py", "humanoid/envs/x1/x1_amp_exp14_config.py",
             "humanoid/envs/x1/x1_amp_exp15_config.py", "humanoid/envs/x1/x1_amp_exp16_config.py",
             "humanoid/envs/x1/x1_amp_exp17_config.py", "humanoid/envs/x1/x1_amp_exp18_config.py",
             "humanoid/envs/x1/x1_amp_exp19_config.py", "humanoid/envs/x1/x1_amp_exp110_config.py",
             "humanoid/envs/x1/x1_amp_exp111_config.py", "humanoid/envs/x1/x1_amp_exp112_config.py",
             "humanoid/envs/x1/x1_amp_exp113_config.py", "humanoid/envs/x1/x1_amp_exp114_common_config.py",
             "humanoid/envs/x1/x1_amp_exp114_config.py", "humanoid/envs/x1/x1_amp_exp115_config.py",
             "humanoid/envs/x1/x1_amp_exp116_config.py", "humanoid/envs/x1/x1_amp_exp117_config.py"]
    for path in paths:
        definitions(path, namespace)
    return namespace


def grounded_library():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MotionLibrary(MOTIONS, URDF, JOINTS, KEYS, dt=.02, history_steps=3,
                             ground_clearance=.002, limit_tolerance=.05)


class Exp117Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = configuration()
        cls.cfg = cls.ns["X1AMPExp117Cfg"]()

    def test_config_preserves_robot_and_uses_absolute_guidance(self):
        source = self.ns["X1AMPExp116Cfg"]
        target = self.ns["X1AMPExp117Cfg"]
        for section in ("env", "control", "commands", "asset", "sim", "noise", "domain_rand", "amp"):
            self.assertIs(getattr(source, section), getattr(target, section))
        scales = self.cfg.rewards.scales
        self.assertEqual((scales.reference_pose, scales.reference_velocity, scales.reference_feet), (1., .5, 1.))
        self.assertEqual((scales.reference_lateral, scales.reference_waist, scales.reference_support), (.5, .3, 1.))
        self.assertEqual((scales.reference_amplitude, scales.net_displacement), (0., 0.))
        self.assertEqual(scales.raw_target_violation, -.1)
        self.assertEqual((self.cfg.env.num_observations, self.cfg.env.num_actions), (6468, 29))
        runner = self.ns["X1AMPExp117CfgPPO"]().runner
        self.assertEqual((runner.experiment_name, runner.max_iterations), ("x1_amp_exp117", 3000))
        self.assertEqual(runner.policy_transfer_source_iteration, 4000)
        self.assertEqual(runner.policy_transfer_source_sha256,
                         "fa382e21547cd33fdc30904d7a1f978e50503541bd8532c999b86d72e3b97e84")

    def test_static_baseline_is_zero_and_full_reference_positive(self):
        cycle = torch.stack((torch.sin(torch.linspace(0, 2*math.pi, 64)),
                             torch.cos(torch.linspace(0, 2*math.pi, 64))), -1)
        scale = torch.full_like(cycle, .1)
        baseline = static_cycle_baseline(cycle, scale)
        median = torch.median(cycle, dim=0).values.expand_as(cycle)
        self.assertTrue(torch.all(absolute_phase_credit(median, cycle, baseline, scale) == 0))
        perfect = absolute_phase_credit(cycle, cycle, baseline, scale)
        self.assertTrue((perfect > 0).all())
        floor = self.cfg.rewards.phase_credit_negative_floor
        wrong = absolute_phase_credit(-cycle, cycle, baseline, scale, negative_floor=floor)
        self.assertLess(float(wrong.min()), 0)
        self.assertGreater(float(wrong.min()), floor-1e-6)
        self.assertLessEqual(float(perfect.max()), 1.)
        velocity = static_velocity_baseline(torch.ones(4, 2)*.5, torch.full((4, 2), .2))
        self.assertEqual(tuple(velocity.shape), (4, 2))

    def test_support_requires_clearance_load_and_total_ground_reaction(self):
        target = torch.tensor([[.002, .03], [.002, .03], [.002, .03]])
        actual = torch.tensor([[.002, .025], [.002, .025], [.2, .025]])
        force = torch.tensor([[290., 0.], [145., 145.], [290., 0.]])
        score = foot_support_score(actual, target, force, torch.full((3,), 300.))
        self.assertGreater(float(score[0]), float(score[1]))
        self.assertGreater(float(score[0]), float(score[2]))
        airborne = foot_support_score(actual, target, torch.zeros(3, 2), torch.full((3,), 300.))
        self.assertEqual(airborne.tolist(), [0., 0., 0.])
        wrong_load = foot_support_score(actual[:1], target[:1], torch.tensor([[0., 290.]]),
                                        torch.full((1,), 300.))[0]
        self.assertLess(float(wrong_load), float(score[0]))

    def test_phase_bank_is_finite_grounded_and_matched_to_command(self):
        lib = grounded_library()
        reference = PhaseReference(lib, torch.zeros(len(JOINTS),
                                    device=lib.device), self.cfg.amp.reference_cycles)
        bank = ForwardPhaseBank(reference, lib, "F1_B3_continuous_walk", 27, 90, .3,
                                lib.fk.lower, lib.fk.upper, phases=64,
                                history_steps=self.cfg.amp.history_steps, dt=.02,
                                clearance=self.cfg.amp.ground_clearance)
        report = bank.report
        self.assertEqual(report["phase_count"], 64)
        self.assertGreaterEqual(report["stance_contact_fraction"], .95)
        self.assertLessEqual(report["clearance_min_m"], .003)
        self.assertAlmostEqual(report["forward_speed_m_s"], .3)
        self.assertTrue(0.4 < report["root_height_mean_m"] < .9)
        self.assertTrue(math.isfinite(report["stance_slip_median_m_s"]))
        states, histories, phases = bank.sample(5)
        self.assertEqual(tuple(states.shape), (5, 13+2*len(JOINTS)))
        self.assertEqual(tuple(histories.shape), (5, self.cfg.amp.history_steps,
                                                  lib.feature_dim))
        self.assertEqual(tuple(phases.shape), (5,))
        self.assertTrue(torch.isfinite(states).all() and torch.isfinite(histories).all())
        torch.testing.assert_close(states[:, 7:10],
                                   torch.tensor([[.3, 0., 0.]]).expand(5, -1))

    def test_rsi_schedule_reference_weight_and_coupling_are_relative_to_parent(self):
        cfg = self.ns["X1AMPExp117Cfg"]()
        class Parent:
            def set_training_iteration(self, iteration): pass
            def _reference_weight(self):
                initial = self.cfg.rewards.reference_initial_weight
                return initial+(1-initial)*self._reference_progress
        ns = dict(torch=torch, X1AMPExp116Env=Parent, absolute_phase_credit=absolute_phase_credit)
        definitions("humanoid/envs/x1/x1_amp_exp117_env.py", ns)
        env = ns["X1AMPExp117Env"](); env.cfg = cfg
        env.phase_bank = object(); env._guidance_evaluation = False
        env._iteration = 4000
        self.assertEqual(env._rsi_probability(), .7)
        for iteration, expected in ((4200, .7), (4300, .6), (4500, .5), (4900, .2), (5000, 0.)):
            env._iteration = iteration
            self.assertAlmostEqual(env._rsi_probability(), expected)
        env.set_training_iteration(4000)
        self.assertAlmostEqual(env._reference_weight(), .25)
        env.set_training_iteration(4300)
        self.assertAlmostEqual(env._reference_weight(), 1.)
        env.commands = torch.tensor([[.3, 0., 0.], [0., 0., 0.]])
        env.base_lin_vel = torch.tensor([[.3, 0., 0.], [0., 0., 0.]])
        self.assertEqual(env._phase_coupling().tolist(), [1., 1.])
        env.base_lin_vel[0, 0] = 0.
        self.assertAlmostEqual(float(env._phase_coupling()[0]), .4)
        env._guidance_evaluation = True
        env.cfg.amp.reference_state_initialization = True
        self.assertEqual(env._rsi_probability(), 1.)

    def test_reward_methods_share_shapes_on_kinematic_fixture(self):
        cfg = self.ns["X1AMPExp117Cfg"]()
        lib = grounded_library()
        reference = PhaseReference(lib, torch.zeros(len(JOINTS), device=lib.device),
                                   self.cfg.amp.reference_cycles)
        feet_fk = URDFKinematics(URDF, lib.dof_names, ["left_ankle_roll_link",
                                                       "right_ankle_roll_link"], "cpu")
        class Parent:
            def _reference_valid(self):
                return torch.ones(self.num_envs, dtype=torch.bool)
            def _reference_weight(self):
                return 1.
        ns = dict(torch=torch, X1AMPExp116Env=Parent, LeggedRobot=Parent,
                  absolute_phase_credit=absolute_phase_credit, foot_support_score=foot_support_score,
                  static_cycle_baseline=static_cycle_baseline,
                  static_velocity_baseline=static_velocity_baseline, sole_heights=__import__(
                      "humanoid.amp.gait_rewards", fromlist=["sole_heights"]).sole_heights,
                  ForwardPhaseBank=ForwardPhaseBank, json=__import__("json"))
        definitions("humanoid/envs/x1/x1_amp_exp117_env.py", ns)
        env = ns["X1AMPExp117Env"]()
        env.cfg, env.device, env.num_envs, env.dt = cfg, "cpu", 2, .02
        env.forward_speed = .3
        env.reference_guide, env.reference_foot_fk = reference, feet_fk
        env.pitch_indices = list(reference.pitch_indices)
        names = list(lib.dof_names)
        env.lateral_indices = [i for i, n in enumerate(names)
                               if any(j in n for j in ("hip_roll", "hip_yaw", "ankle_roll"))]
        env.waist_indices = [i for i, n in enumerate(names) if n.startswith("lumbar_")]
        env.commands = torch.tensor([[.3, 0., 0.], [.3, 0., 0.]])
        env.base_lin_vel = torch.tensor([[.3, 0., 0.], [.3, 0., 0.]])
        env.gait_phase = torch.tensor([.1, .4])
        env.reference_q, env.reference_dq = reference.sample(env.commands[:, 0], env.gait_phase)
        env.reference_feet = feet_fk.key_positions(env.reference_q).flatten(-2)
        env._guidance_feet = lambda: env.reference_feet.clone()
        env.dof_pos = env.reference_q.clone()
        env.dof_vel = env.reference_dq.clone()
        env.sole_height = torch.tensor([[.002, .03], [.002, .03]])
        env.contact_forces = torch.zeros(2, 2, 3); env.contact_forces[:, 0, 2] = 290.
        env.total_mass = torch.full((2, 1), 30.)
        env.feet_indices = torch.tensor([0, 1])
        meshes = {lib.key_body_names[i]: mesh for i, mesh in lib.grounding.feet}
        env.sole_vertices = [meshes["left_ankle_roll_link"], meshes["right_ankle_roll_link"]]
        env.hard_dof_limits = torch.stack((lib.fk.lower, lib.fk.upper), -1)
        env.pd_target_raw = env.dof_pos.clone()
        env._build_static_baseline()
        values = {"pose": env._reward_reference_pose(),
                  "velocity": env._reward_reference_velocity(),
                  "feet": env._reward_reference_feet(),
                  "lateral": env._reward_reference_lateral(),
                  "waist": env._reward_reference_waist(),
                  "support": env._reward_reference_support(),
                  "raw_target": env._reward_raw_target_violation()}
        for name, value in values.items():
            self.assertEqual(tuple(value.shape), (2,), name)
            self.assertTrue(torch.isfinite(value).all(), name)
        self.assertTrue((values["pose"] > 0).all())
        self.assertTrue((values["support"] > 0).all())

    def test_reset_uses_phase_bank_for_forward_and_standing_otherwise(self):
        cfg = self.ns["X1AMPExp117Cfg"]()
        lib = grounded_library()
        reference = PhaseReference(lib, torch.zeros(len(JOINTS), device=lib.device),
                                   self.cfg.amp.reference_cycles)
        bank = ForwardPhaseBank(reference, lib, "F1_B3_continuous_walk", 27, 90, .3,
                                lib.fk.lower, lib.fk.upper, phases=64,
                                history_steps=self.cfg.amp.history_steps, dt=.02,
                                clearance=self.cfg.amp.ground_clearance)
        clip = lib.names.index("F1_B1_stand_to_walk")
        standing = lib.state(torch.tensor([clip]), torch.tensor([11/float(lib.fps[clip])]))[0]
        class FakeReset:
            def reset_idx(self, env_ids):
                self.standing_commands.requested[env_ids] = self._reset_commands
                self.dof_pos[env_ids] = self._reset_reference[:, 13:42]
                self.dof_vel[env_ids] = self._reset_reference[:, 42:71]
                self.commands[env_ids] = self.standing_commands.applied()[env_ids]
        ns = dict(torch=torch, X1AMPExp116Env=object, LeggedRobot=FakeReset,
                  absolute_phase_credit=absolute_phase_credit, foot_support_score=foot_support_score,
                  static_cycle_baseline=static_cycle_baseline,
                  static_velocity_baseline=static_velocity_baseline, sole_heights=__import__(
                      "humanoid.amp.gait_rewards", fromlist=["sole_heights"]).sole_heights,
                  ForwardPhaseBank=ForwardPhaseBank, json=__import__("json"))
        definitions("humanoid/envs/x1/x1_amp_exp117_env.py", ns)
        env = ns["X1AMPExp117Env"]()
        env.cfg, env.device, env.num_envs = cfg, "cpu", 4
        env.motion_library, env.reference_guide, env.phase_bank = lib, reference, bank
        env.standing_state = standing
        env.hard_dof_limits = torch.stack((lib.fk.lower, lib.fk.upper), -1)
        env.standing_commands = StandingCommands(4, "cpu", 75)
        env.reset_standing = torch.zeros(4, dtype=torch.bool)
        env.obs_history = [torch.zeros(4, 66, 98)]
        env.amp_history = torch.zeros(4, self.cfg.amp.history_steps, lib.feature_dim)
        env.zero_steps = torch.zeros(4, dtype=torch.long); env.stop_hold_steps = torch.zeros(4, dtype=torch.long)
        env.previous_moving = torch.zeros(4, dtype=torch.bool); env.stop_pending = torch.zeros(4, dtype=torch.bool)
        env.stand_gate = torch.zeros(4)
        env.gait_phase = torch.zeros(4); env.previous_phase_command = torch.zeros(4)
        env.commands = torch.zeros(4, 3); env.dof_pos = standing[13:42].repeat(4, 1)
        env.dof_vel = torch.zeros_like(env.dof_pos)
        env._guidance_evaluation = True
        env._draw_commands = lambda count: torch.tensor([[.3, 0., 0.]]).expand(count, -1).clone()
        states, histories, phases = bank.sample(4)
        bank.sample = lambda count: (states.clone(), histories.clone(), phases.clone())
        env.reset_idx(torch.arange(4))
        self.assertFalse(env.reset_standing.any())
        torch.testing.assert_close(env.dof_pos, states[:, 13:42])
        torch.testing.assert_close(env.dof_vel, states[:, 42:71])
        torch.testing.assert_close(env.amp_history, histories)
        torch.testing.assert_close(env.gait_phase, phases)
        torch.testing.assert_close(env.commands[:, 0], torch.full((4,), .3))
        env.cfg.amp.reference_state_initialization = False
        env._guidance_evaluation = False
        env.reset_idx(torch.arange(4))
        self.assertTrue(env.reset_standing.all())
        self.assertTrue((env.standing_commands.remaining == 75).all())
        self.assertTrue(torch.isfinite(env.dof_pos).all())
        self.assertTrue((env.dof_pos >= env.hard_dof_limits[:, 0]-1e-6).all())
        self.assertTrue((env.dof_pos <= env.hard_dof_limits[:, 1]+1e-6).all())

    def test_negative_credit_cannot_reward_an_early_fall(self):
        cfg = self.cfg.rewards
        dt, steps, style_mix = 1/50, int(20*50), .5
        guidance_weights = (cfg.scales.reference_pose+cfg.scales.reference_velocity
                            + cfg.scales.reference_feet+cfg.scales.reference_lateral
                            + cfg.scales.reference_waist)
        worst = (abs(cfg.phase_credit_negative_floor)*guidance_weights
                 + abs(cfg.scales.raw_target_violation)*cfg.raw_target_max_sum)*dt*steps*style_mix
        self.assertGreater(abs(cfg.non_timeout_termination_penalty), worst)

    def test_reward_transfer_contract_is_explicit(self):
        runner = self.ns["X1AMPExp117CfgPPO"]().runner
        self.assertEqual(runner.warm_start_mode, "policy_amp_fresh_critic")
        self.assertTrue(runner.require_reward_transfer)
        self.assertFalse(runner.require_policy_transfer)
        algorithm = self.ns["X1AMPExp117CfgPPO"]().algorithm
        self.assertEqual((algorithm.policy_transfer_warmup_iterations,
                          algorithm.policy_transfer_warmup_max_lr), (200, 1e-5))
        self.assertEqual((algorithm.min_learning_rate, algorithm.max_learning_rate), (1e-5, 3e-5))


if __name__ == "__main__":
    unittest.main()
