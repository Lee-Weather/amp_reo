"""CPU checks of control semantics, buffer boundaries and metadata compatibility."""
import ast
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
from humanoid.amp.exp26_swing import ClearanceRuns
from humanoid.amp.nominal_startup import configure_nominal_probe


def final_method(name, **globals_):
    """Execute a production method without importing the unavailable Gym runtime."""
    path = Path(__file__).resolve().parents[1]/'humanoid/envs/x1/x1_amp_env.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'X1AMPFinalEnv')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(torch=torch, **globals_)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Exp11Tests(unittest.TestCase):
    def test_nominal_probe_uses_the_real_importer_friction_setter(self):
        cfg=SimpleNamespace(amp=SimpleNamespace(),domain_rand=SimpleNamespace(),
                            sim=SimpleNamespace(physx=SimpleNamespace()))
        configure_nominal_probe(cfg)
        path=Path(__file__).resolve().parents[1]/'humanoid/envs/base/legged_robot.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LeggedRobot')
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_process_rigid_shape_props')
        ns=dict(torch=torch,torch_rand_float=lambda lo,hi,shape,device: lo+(hi-lo)*torch.rand(shape))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
        env=SimpleNamespace(cfg=cfg,num_envs=1,env_frictions=torch.zeros(1,1))
        props=[SimpleNamespace(friction=1.,restitution=.5) for _ in range(3)]
        ns['_process_rigid_shape_props'](env,props,0)
        self.assertTrue(all(abs(float(p.friction)-.8)<1e-6 and p.restitution==0 for p in props))
        self.assertAlmostEqual(float(env.env_frictions[0]),.8,places=6)

    def test_single_environment_has_no_nominal_cohort_to_inspect(self):
        calls = []
        shape = SimpleNamespace(friction=.8, contact_offset=.001, rest_offset=0.)
        env = SimpleNamespace(envs=[0, 1], actor_handles=[0, 1],
            gym=SimpleNamespace(get_actor_rigid_shape_properties=lambda *args: calls.append(args) or [shape]))
        verify = final_method('_verify_nominal_startup_shapes')
        self.assertEqual(verify(env, 0), [])
        self.assertEqual(calls, [])
        self.assertEqual([row['env_id'] for row in verify(env, 2)], [0, 1])
        shape.friction = .65
        with self.assertRaisesRegex(ValueError, 'Nominal cohort'):
            verify(env, 2)

    def test_production_mixed_reset_clears_all_foot_event_histories(self):
        count, joints, frames, width = 4, 29, 66, 98
        tracker = ClearanceRuns(count, 'cpu', .02)
        tracker.seconds[:] = .1
        tracker.counted[:] = True
        commands = SimpleNamespace(**{key: torch.zeros(count) for key in ('active_goal', 'start', 'value')},
            requested=torch.zeros(count, 3), reset=lambda *_: None, applied=lambda: torch.zeros(count, 3))
        bank = dict(states=torch.zeros(2, 71), phases=torch.zeros(2), histories=torch.zeros(2, frames, width),
                    amp_states=torch.zeros(2, 3, 71))
        env = SimpleNamespace(device='cpu', gait_curriculum=SimpleNamespace(weights={'rsi_probability': .5}),
            evaluation_reset_mode=None, bank=bank, standing_state=torch.zeros(71),
            cfg=SimpleNamespace(amp=SimpleNamespace(standing_joint_noise=0., ground_clearance=.002, history_steps=3),
                                commands=SimpleNamespace(native_speed=.595)),
            hard_dof_limits=torch.tensor([[-1., 1.]]*joints),
            reference_guide=SimpleNamespace(sole_height=lambda state: torch.zeros(len(state), 2),
                                           nearest_phase=lambda q, dq: torch.zeros(len(q))),
            _draw_commands=lambda n: torch.zeros(n, 3), standing_commands=commands,
            reset_standing=torch.zeros(count, dtype=torch.bool), commands=torch.zeros(count, 3),
            obs_history=[torch.zeros(count, width) for _ in range(frames)],
            amp_history=torch.zeros(count, 3, 86), amp_valid_history=torch.zeros(count, 3, dtype=torch.bool),
            motion_library=SimpleNamespace(features=lambda state: torch.zeros(len(state), 86)),
            gait_phase=torch.zeros(count), completed_cycles=torch.zeros(count), swing_air_tracker=tracker)
        # Force a standing/RSI mixture through the actual flattened reset path.
        env._adjust_reset_states=lambda ids, standing, phase, state: (torch.tensor([True, False]), phase, state)
        for name in ('actions', 'last_actions', 'last_last_actions', 'last_dof_vel', 'dof_vel'):
            setattr(env, name, torch.zeros(count, joints))
        for name in ('zero_steps', 'stop_hold_steps', 'previous_moving', 'stop_pending', 'stand_gate'):
            setattr(env, name, torch.zeros(count))
        reset = final_method('reset_idx', LeggedRobot=SimpleNamespace(reset_idx=lambda *_: None))
        reset(env, torch.tensor([0, 2]))
        torch.testing.assert_close(tracker.seconds[[0, 2]], torch.zeros(2, 2))
        self.assertFalse(tracker.counted[[0, 2]].any())
        torch.testing.assert_close(tracker.seconds[[1, 3]], torch.full((2, 2), .1))

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
