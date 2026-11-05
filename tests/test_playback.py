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
import numpy as np

from humanoid.amp.playback import command_at, inject_command, speed_profile
from humanoid.amp.replay import ReplayRecorder
from humanoid.amp.transition_trace import TransitionTrace, load_trace, compare_traces, validate_replay_source
from humanoid.amp.wbt_teacher import WBTTeacher, permutation, student_action, termination
from humanoid.amp.teacher_dataset import canonical_action, validate_dataset, audit_conditions, audit_exact_label_conflicts
from humanoid.amp.reference_condition import ReferenceCondition, causal_speed_envelope
from humanoid.amp.student_distillation import (SCHEMA, SOURCE_SHA, validate_plan, freeze_features,
    frozen_digest, state_digest, checked_backward, fit_metrics, save_checkpoint,
    restore_checkpoint, trusted_load, checkpoint_directory)


class DistillationSafetyTests(unittest.TestCase):
    def test_control_input_download_requires_https_and_exact_sha(self):
        import base64
        import hashlib
        from unittest.mock import patch
        from humanoid.scripts.probe_b1_platform import download
        encoded=base64.urlsafe_b64encode(b'https://example.invalid/private?signature=hidden').decode()
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'abc')):
            self.assertEqual(download(encoded,hashlib.sha256(b'abc').hexdigest()),b'abc')
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'abc')):
            with self.assertRaisesRegex(RuntimeError,'URL withheld'):
                download(encoded,'0'*64)
        with patch('urllib.request.urlopen') as opener:
            with self.assertRaisesRegex(RuntimeError,'URL withheld'):
                download(base64.urlsafe_b64encode(b'http://example.invalid').decode(),'0'*64)
            opener.assert_not_called()

    def test_platform_checkpoint_directory_rejects_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'input_source.pt'
            source.write_bytes(b'toy source')
            self.assertEqual(checkpoint_directory(root/'new_metadata',root,source),root.resolve())
            (root/'model_50.pt').write_bytes(b'old output')
            with self.assertRaisesRegex(ValueError,'only the verified'):
                checkpoint_directory(root/'new_metadata',root,source)

    def test_private_platform_bundle_is_hash_checked_and_cannot_overwrite(self):
        import hashlib
        import tarfile
        from humanoid.scripts.train_b1_platform import unpack_inputs
        blob=io.BytesIO()
        with tarfile.open(fileobj=blob,mode='w:gz') as archive:
            item=tarfile.TarInfo('private/reference.npz'); item.size=3
            archive.addfile(item,io.BytesIO(b'abc'))
        plan=dict(inputs=dict(reference=dict(path='private/reference.npz',sha256=hashlib.sha256(b'abc').hexdigest())))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            unpack_inputs(blob.getvalue(),plan,root)
            self.assertEqual((root/'private/reference.npz').read_bytes(),b'abc')
            with self.assertRaises(FileExistsError):
                unpack_inputs(blob.getvalue(),plan,root)
            plan['inputs']['reference']['sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                unpack_inputs(blob.getvalue(),plan,root)

    def plan(self):
        return dict(schema=SCHEMA, seed=5, trainable='actor_only', independent_validation_episodes=0,
            acceptance=dict(offline_raw_target_rmse_rad_max=.06, offline_rmse_relative_max=.7,
                closed_loop_steps=384, closed_loop_mean_joint_rmse_rad_max=.12,
                closed_loop_max_root_error_m=.35, zero_early_termination=True),
            budget=dict(updates=200, batch_size=64, learning_rate=1e-4, max_grad_norm=1.,
                max_wall_seconds=600, save_interval=50, max_raw_target_shift_rms_rad=.20),
            inputs={name:dict(path=name, sha256=SOURCE_SHA if name == 'checkpoint' else 'a'*64)
                    for name in ('checkpoint', 'dataset', 'profile', 'teacher_lock', 'teacher_bundle', 'reference')})

    def toy(self):
        torch.manual_seed(5)
        model = torch.nn.Module()
        model.actor = torch.nn.Linear(4, 2)
        model.feature = torch.nn.Linear(3, 4)
        model.log_std = torch.nn.Parameter(torch.zeros(2))
        freeze_features(model)
        return model

    def test_budget_and_input_contract_cannot_silently_expand(self):
        import copy
        plan = self.plan()
        validate_plan(plan)
        for section, key, value in ((None, 'seed', 7), ('budget', 'updates', 201),
                                    ('budget', 'learning_rate', .01), (None, 'trainable', 'all')):
            changed = copy.deepcopy(plan)
            (changed if section is None else changed[section])[key] = value
            with self.assertRaises(ValueError):
                validate_plan(changed)

    def test_backward_has_finite_gradients_and_preserves_frozen_weights_without_step(self):
        model = self.toy()
        before = state_digest(model.state_dict())
        loss = model.actor(torch.ones(3, 4)).square().mean()
        norm = checked_backward(model, loss)
        self.assertGreater(norm, 0)
        self.assertEqual(before, state_digest(model.state_dict()))
        self.assertIsNone(model.feature.weight.grad)
        self.assertIsNone(model.log_std.grad)
        with self.assertRaisesRegex(ValueError, 'Nonfinite loss'):
            checked_backward(model, loss*float('nan'))

    def test_nonfinite_gradient_with_finite_loss_is_rejected(self):
        model = self.toy()
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.zero_()
        loss = model.actor(torch.ones(1, 4)).sqrt().sum()  # finite zero value, infinite derivative
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, 'nonfinite actor gradient'):
            checked_backward(model, loss)

    def test_resume_restores_optimizer_sampling_and_exact_next_update(self):
        plan = self.plan()
        model = self.toy()
        frozen = frozen_digest(model)
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(5)
        inputs = torch.arange(40, dtype=torch.float32).reshape(10, 4)/40
        def update(m, opt, gen):
            indices = torch.randperm(10, generator=gen)[:3]
            opt.zero_grad(set_to_none=True)
            checked_backward(m, m.actor(inputs[indices]).square().mean())
            opt.step()
            return indices
        update(model, optimizer, generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'student.pt'
            save_checkpoint(path, model, optimizer, generator, 1, plan,
                dict(config={}, metadata={}), frozen, dict(step=1))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, generator, 1, plan,
                    dict(config={}, metadata={}), frozen, {})
            import hashlib
            checkpoint = trusted_load(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(checkpoint['metadata']['schema_version'], SCHEMA)
            self.assertFalse(checkpoint['amp_resume_compatible'])
            expected_indices = update(model, optimizer, generator)
            expected_digest = state_digest(model.state_dict())
            resumed = self.toy()
            resumed_optimizer = torch.optim.Adam(resumed.actor.parameters(), lr=1e-4)
            resumed_generator = torch.Generator()
            step, _ = restore_checkpoint(checkpoint, resumed, resumed_optimizer, resumed_generator, plan, frozen)
            self.assertEqual(step, 1)
            torch.testing.assert_close(expected_indices, update(resumed, resumed_optimizer, resumed_generator))
            self.assertEqual(expected_digest, state_digest(resumed.state_dict()))
            self.assertEqual(frozen, frozen_digest(resumed))
            checkpoint['plan'] = dict(plan, seed=7)
            with self.assertRaisesRegex(ValueError, 'contract'):
                restore_checkpoint(checkpoint, resumed, resumed_optimizer, resumed_generator, plan, frozen)

    def test_hash_failure_precedes_deserialization(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.pt'
            path.write_bytes(b'untrusted content')
            with patch('humanoid.amp.student_distillation.torch.load') as loader:
                with self.assertRaisesRegex(ValueError, 'refusing deserialization'):
                    trusted_load(path, 'a'*64)
                loader.assert_not_called()

    def test_clipped_fit_cannot_hide_raw_policy_error(self):
        result = fit_metrics(torch.tensor([[100.]]), torch.tensor([[1.]]), torch.zeros(1),
            torch.tensor([[-1., 1.]]), 1., 5., torch.zeros(1, 1))
        self.assertEqual(result['effective_target_rmse_rad'], 0)
        self.assertEqual(result['raw_target_rmse_rad'], 99)
        self.assertEqual(result['action_clip_fraction'], 1)


class ReferenceConditionTests(unittest.TestCase):
    def profile(self, path):
        progress=np.linspace(0,1,5,dtype=np.float32)
        features=np.stack((np.sin(np.pi*progress),np.cos(np.pi*progress)),-1)
        commands=np.zeros((5,3),dtype=np.float32);commands[:,0]=np.arange(5)*.1
        np.savez(path,progress=progress,features=features,commands=commands,
            metadata=np.array(json.dumps(dict(schema='x1_reference_progress_condition_v1',policy_dt=.02))))
        return ReferenceCondition(path)

    def test_finite_progress_distinguishes_endpoints_and_rejects_implicit_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            profile=self.profile(Path(directory)/'profile.npz')
            features=np.asarray([profile.sample(i)[1] for i in range(5)])
            np.testing.assert_allclose(np.arctan2(np.maximum(features[:,0],0),features[:,1])/np.pi,np.linspace(0,1,5),atol=1e-6)
            self.assertFalse(np.allclose(features[0],features[-1]))
            with self.assertRaisesRegex(ValueError,'no implicit'):
                profile.sample(5)

    def test_online_condition_changes_only_latest_condition_channels_and_checks_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            profile=self.profile(Path(directory)/'profile.npz')
            env=SimpleNamespace(device='cpu',dt=.02,episode_length_buf=torch.tensor([2]),
                cfg=SimpleNamespace(env=SimpleNamespace(num_single_obs=14)),
                commands=torch.zeros(1,3),commands_scale=torch.tensor([2.,2.,.25]),
                obs_buf=torch.arange(42,dtype=torch.float32)[None],
                privileged_obs_buf=torch.zeros(1,17),obs_history=deque([torch.full((1,14),float(i)) for i in range(3)]))
            before=env.obs_buf.clone();past=[v.clone() for v in env.obs_history]
            profile.apply(env,2)
            torch.testing.assert_close(env.obs_buf[:,:-14],before[:,:-14])
            torch.testing.assert_close(env.obs_buf[:,-11:-2],before[:,-11:-2])
            for a,b in zip(list(env.obs_history)[:-1],past[:-1]):torch.testing.assert_close(a,b)
            torch.testing.assert_close(env.obs_buf[:,-2:],torch.tensor(profile.sample(2)[1])[None])
            torch.testing.assert_close(env.obs_buf[:,-14:-11],torch.tensor([[.4,0.,0.]]))
            with self.assertRaisesRegex(ValueError,'actual episode'):
                profile.apply(env,3)

    def test_planned_speed_envelope_does_not_consult_future_reference_samples(self):
        original=np.array([0.,0.,.2,.4,.6,0.])
        changed=original.copy();changed[4:]=10.
        a=causal_speed_envelope(original,window=3,deadband=0)
        b=causal_speed_envelope(changed,window=3,deadband=0)
        np.testing.assert_array_equal(a[:4],b[:4])
        np.testing.assert_allclose(a[:4],[0.,0.,.2/3,.2],atol=1e-7)


class TeacherDatasetTests(unittest.TestCase):
    def fixture(self):
        count,width,frames=4,14,3
        actions=np.arange(1,count+1,dtype=np.float32)[:,None]*.1
        previous=np.vstack((np.zeros((1,1)),actions[:-1]))
        q=np.arange(count)[:,None]*.01
        history=np.zeros((frames,width));before=[];after=[]
        def frame(i,previous_action):
            return np.r_[.6,0.,0.,0.,0.,0.,0.,0.,-1.,i*.01,.02,previous_action,0.,0.]
        history[-1]=frame(0,0.)
        for i in range(count):
            before.append(history.copy().reshape(-1))
            history[:-1]=history[1:]
            history[-1]=frame(i+1,float(actions[i,0]))
            after.append(history.copy().reshape(-1))
        meta=dict(schema='x1_teacher_student_dataset_v1',frame_width=width,frame_stack=frames,
            dof_names=['j'],policy_dt=.02,observation_clip=100.,default_dof_pos=[0.],action_scale=.25,
            hard_limits=[[-1.,1.]],command_scale=[2.,2.,.25],observation_scales=dict(ang_vel=.25,dof_pos=1.,dof_vel=.1))
        return dict(metadata=np.array(json.dumps(meta)),student_observations=np.asarray(before),
            next_student_observations=np.asarray(after),pre_q=q,pre_dq=np.full((count,1),.2),
            pre_roots=np.tile(np.array([0.,0.,.6,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]),(count,1)),
            previous_actions=previous,applied_actions=actions,label_actions=actions.copy(),effective_targets=.25*actions,
            applied_commands=np.tile([.3,0.,0.],(count,1)),phase_features=np.zeros((count,2)),
            time_s=np.arange(count)*.02,done=np.zeros(count),
            reference_commands=np.tile([.3,0.,0.],(count,1)),student_phase=np.zeros(count),reference_native_phase=np.zeros(count))

    def test_dataset_checks_real_history_lag_and_effective_target_labels(self):
        data=self.fixture()
        self.assertTrue(validate_dataset(data)['causal_history_verified'])
        data['student_observations'][0,-3]=data['applied_actions'][0,0]
        with self.assertRaisesRegex(ValueError,'pre-action'):
            validate_dataset(data)
        data=self.fixture();data['label_actions'][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'Label does not encode'):
            validate_dataset(data)
        data=self.fixture();data['next_student_observations'][0,0]=1.
        with self.assertRaisesRegex(ValueError,'next pre-action'):
            validate_dataset(data)

    def test_dataset_stops_at_terminal_boundary_and_reports_condition_disagreement(self):
        data=self.fixture();data['done'][1]=1
        with self.assertRaisesRegex(ValueError,'reset boundary'):
            validate_dataset(data)
        data=self.fixture();data['done'][-1]=1;data['next_student_observations'][-1]=0
        self.assertEqual(validate_dataset(data)['terminal_rows'],1)
        data=self.fixture();data['reference_commands'][0,0]=0
        self.assertEqual(audit_conditions(data)['command_moving_while_reference_quiet_rows'],1)
        self.assertEqual(audit_conditions(data)['command_motion_category_disagreement_fraction'],.25)

    def test_canonical_behavior_action_encodes_bounded_target(self):
        action,target=canonical_action(np.array([10.]),np.array([0.]),.25,18.,np.array([[-1.,1.]]))
        np.testing.assert_allclose(target,[1.])
        np.testing.assert_allclose(action,[4.])
        np.testing.assert_allclose(.25*action,target)

    def test_identical_inputs_with_conflicting_targets_are_not_accepted_as_unambiguous(self):
        data=self.fixture()
        self.assertEqual(audit_exact_label_conflicts(data)['duplicate_input_groups'],0)
        data['student_observations'][1]=data['student_observations'][0]
        self.assertEqual(audit_exact_label_conflicts(data)['conflicting_duplicate_groups'],1)


class WBTTeacherTests(unittest.TestCase):
    def test_observation_rotation_joint_mapping_and_action_conversion(self):
        teacher=object.__new__(WBTTeacher)
        teacher.names=['j'+str(i) for i in range(29)]
        teacher.actor_names=teacher.names[::-1]
        teacher.to_actor=permutation(teacher.names,teacher.actor_names)
        teacher.default=np.arange(29,dtype=np.float32)*.01
        teacher.meta=dict(body_names=['base_link'],action_scale=.5)
        teacher.frames=2
        quat=np.array([0.,0.,np.sqrt(.5),np.sqrt(.5)])
        teacher.data=dict(joint_pos=np.tile(np.arange(29),(2,1)),joint_vel=np.tile(np.arange(29)+30,(2,1)),
            body_pos_w=np.array([[[1.,0.,.6]]]*2),body_quat_w=np.array([[quat[[3,0,1,2]]]]*2))
        root=np.array([0.,0.,.6,*quat,0.,1.,0.,0.,0.,2.])
        last=np.arange(29,dtype=np.float32)*.03
        obs=teacher.observation(0,root,teacher.default+.2,np.ones(29),last,teacher.names)
        np.testing.assert_allclose(obs[:29],np.arange(29)[::-1])
        np.testing.assert_allclose(obs[58:61],[0.,-1.,0.],atol=1e-7)
        np.testing.assert_allclose(obs[61:67],[1.,0.,0.,1.,0.,0.],atol=1e-7)
        np.testing.assert_allclose(obs[67:70],[1.,0.,0.],atol=1e-7)
        np.testing.assert_allclose(obs[73:102],.2,atol=1e-7)
        np.testing.assert_array_equal(obs[-29:],last)
        target=teacher.target(last,teacher.names)
        np.testing.assert_allclose(target,teacher.default+.5*last[::-1])
        action,raw=student_action(target,teacher.default,.25,18.)
        np.testing.assert_allclose(action,2*last[::-1],atol=1e-6)
        with self.assertRaisesRegex(ValueError,'outside clip'):
            teacher.observation(2,root,teacher.default,np.ones(29),last,teacher.names)
        with self.assertRaisesRegex(ValueError,'unique'):
            permutation(['a','a'],['a','b'])

    def test_teacher_replay_has_active_termination_and_action_clipping(self):
        names=['base_link','left_ankle_roll_link','right_ankle_roll_link','left_wrist_roll_link','right_wrist_roll_link']
        bodies=np.zeros((5,3));root=np.array([0.,0.,.6,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.])
        args=(np.array([0.,0.,.6]),np.array([1.,0.,0.,0.]),bodies,bodies.copy(),names)
        self.assertIsNone(termination(root,*args))
        root[2]=.2
        self.assertEqual(termination(root,*args),'base_below_0.3m')
        root[2]=.6;bodies[1,2]=.3
        self.assertEqual(termination(root,*args),'left_ankle_roll_link_height_error_above_0.25m')
        clipped,raw=student_action(np.array([10.]),np.array([0.]),.25,18.)
        self.assertEqual(clipped[0],18.)
        self.assertEqual(raw[0],40.)


class TransitionTraceTests(unittest.TestCase):
    def fixture(self):
        root = np.array([0., 0., .6, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        trace = TransitionTrace(dict(engine='fixture', checkpoint_sha256='abc', dof_names=['a', 'b'],
            num_observations=3, policy_dt=.02), root, np.zeros(2), np.zeros(2), np.zeros(3))
        for _ in range(3):
            trace.append(np.zeros(3), np.zeros(2), root, np.zeros(2), np.zeros(2), np.zeros((2, 3)))
        return trace

    def test_trace_roundtrip_preserves_initial_history_and_never_overwrites(self):
        trace = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.npz'
            trace.save(path)
            data = load_trace(path)
            np.testing.assert_array_equal(data['observations'], trace.arrays()['observations'])
            with self.assertRaises(FileExistsError):
                trace.save(path)
        report = compare_traces(data, data)
        self.assertIsNone(report['first_state_divergence_s'])
        self.assertEqual(report['compared_steps'], 3)

    def test_comparison_finds_first_divergence_and_handles_quaternion_sign(self):
        left = self.fixture().arrays()
        right = {key: value.copy() for key, value in left.items()}
        right['roots'][:, 3:7] *= -1
        right['q'][1, 0] = .06
        right['roots'][2, 0] = .02
        report = compare_traces(left, right)
        self.assertAlmostEqual(report['first_state_divergence_s'], .04)
        self.assertAlmostEqual(report['first_exceeded_s']['position_m'], .06)
        self.assertEqual(report['maxima']['orientation_rad'], 0.)

    def test_unmatched_actions_initial_state_and_checkpoint_are_rejected(self):
        left = self.fixture().arrays()
        for key, index in [('actions', (0, 0)), ('initial_q', 0)]:
            right = {k: v.copy() for k, v in left.items()}
            right[key][index] = .1
            with self.assertRaises(ValueError):
                compare_traces(left, right)
        with self.assertRaisesRegex(ValueError, 'different checkpoint'):
            validate_replay_source(left, 'wrong', ['a', 'b'], .02, 3)

    def test_trace_rejects_nonfinite_data_and_reset_splicing(self):
        trace = self.fixture()
        trace.rows[-1]['done'] = np.array(True)
        with self.assertRaisesRegex(ValueError, 'first reset'):
            trace.append(np.zeros(3), np.zeros(2), np.zeros(13), np.zeros(2), np.zeros(2), np.zeros((2, 3)))
        trace.rows[0]['actions'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'actions'):
            trace.arrays()

    def test_moving_start_is_rejected_until_velocity_reference_is_verified(self):
        data = self.fixture().arrays()
        validate_replay_source(data, 'abc', ['a', 'b'], .02, 3)
        data['initial_root'][7] = .2
        with self.assertRaisesRegex(ValueError, 'stationary'):
            validate_replay_source(data, 'abc', ['a', 'b'], .02, 3)


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
                 "reference_phase": torch.tensor([.75]), "reference_q": torch.tensor([[.7]]),
                 "reference_dq": torch.tensor([[.1]]),
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
            self.assertAlmostEqual(float(row["style_score"]), .5)
            self.assertAlmostEqual(float(row["style_reward"]), .02)
            self.assertEqual(float(row["command_requested"]), .6)
            self.assertEqual(float(row["foot_force_l"]), 20.)
            self.assertEqual(float(row["foot_force_r"]), 30.)
            self.assertAlmostEqual(float(row["pos_des_raw_joint"]), .4)
            self.assertEqual(int(row["clip_count"]), 7)
            self.assertAlmostEqual(float(row["reference_phase"]), .75)
            self.assertAlmostEqual(float(row["reference_q_joint"]), .7)
            self.assertAlmostEqual(float(row["reference_dq_joint"]), .1)
            metadata = json.loads((Path(tmp)/"replay_metadata.json").read_text())
            self.assertEqual(metadata["timeouts"], 1)
            self.assertEqual(metadata["captured_steps"], 1)
            self.assertFalse((Path(tmp)/"play_output.mp4").exists())
            with self.assertRaises(FileExistsError):
                ReplayRecorder(env, runner, tmp, render=False)
