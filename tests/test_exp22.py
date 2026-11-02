"""Actual source-loss gradients, PPO rollback, sampling and resume contracts."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.anchored_runner import AnchoredAMPOnPolicyRunner
from humanoid.amp.motion import quat_inverse_rotate
from test_exp21 import SOURCE,fixture21,configuration21
from test_exp117 import definitions
from preflight_reference_transfer import dictionary
from humanoid.amp.gait_curriculum import GaitCurriculum
from humanoid.amp.exp20_commands import SmoothCommands

ROOT=Path(__file__).resolve().parents[1]

def fixture22(source):
    env,cfg=fixture21(source)
    ns=configuration21();definitions('humanoid/envs/x1/x1_amp_exp22_config.py',ns)
    cfg.update(dictionary(ns['X1AMPExp22CfgPPO']()))
    bank=json.loads(Path(os.environ.get('EXP22_STATE_BANK',str(ROOT/'humanoid/amp/reference_data/exp22_policy_state_bank.json'))).read_text())
    env.source_anchor_observations=torch.tensor(bank['histories']+bank['quiet_histories']).flatten(1)
    env.checkpoint_training_metadata=lambda:dict(curriculum='anchored_policy_state_native_v1',
        reference_source_sha256=source['metadata']['training']['reference_source_sha256'])
    return env,cfg,bank

class AnchoredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg,_=fixture22(self.source)
        r=AnchoredAMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_bank_identity_accepts_simulator_float_but_rejects_wrong_rate(self):
        ns=dict(X1AMPExp21Env=object,GaitCurriculum=GaitCurriculum)
        definitions('humanoid/envs/x1/x1_amp_exp22_env.py',ns)
        validate=ns['validate_policy_bank'];bank=dict(dof_names=['a'],control_dt=.02)
        validate(bank,['a'],float(torch.tensor(.001,dtype=torch.float32))*20)
        with self.assertRaises(ValueError):validate(bank,['a'],.01)
        with self.assertRaises(ValueError):validate(bank,['b'],.02)

    def test_sampling_gain_is_in_distribution_and_log_probability(self):
        r=self.runner();obs=r.env.source_anchor_observations
        reduced=r.actor_critic.distribution(obs)
        r.actor_critic.exploration_gain=1.
        original=r.actor_critic.distribution(obs)
        torch.testing.assert_close(reduced.scale,original.scale*.5)
        torch.testing.assert_close(reduced.loc,original.loc)
        action=reduced.sample()
        torch.testing.assert_close(reduced.log_prob(action),torch.distributions.Normal(original.loc,original.scale*.5).log_prob(action))

    def test_source_loss_gradient_actually_pulls_actor_back(self):
        r=self.runner();r.current_learning_iteration=50;r._freeze_inference()
        obs=r.env.source_anchor_observations
        with torch.no_grad():r.actor_critic.actor[-1].bias.add_(.02)
        before=float(r.policy_regularization_loss(obs))
        r.optimizer.zero_grad();loss=r.policy_regularization_loss(obs);loss.backward()
        self.assertGreater(float(r.actor_critic.actor[-1].bias.grad.norm()),0)
        torch.nn.utils.clip_grad_norm_(r.actor_critic.parameters(),1.)
        r.optimizer.step()
        self.assertLess(float(r.policy_regularization_loss(obs)),before)
        self.assertTrue(all(p.grad is None for p in r.policy_anchor.parameters()))

    def test_ppo_guard_rolls_back_model_and_optimizer(self):
        r=self.runner();r.current_learning_iteration=50
        batch=r.collect_rollout();before=copy.deepcopy(r.actor_critic.state_dict())
        r.alg_cfg['source_max_pd_rms']=0.
        metrics=r.update_policy(batch)
        self.assertEqual(metrics['source_guard_rejected'],1.)
        self.assertFalse(r.optimizer.state)
        for k,v in before.items():torch.testing.assert_close(v,r.actor_critic.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(metrics['source_bank_pd_rms'],0.)

    def test_feature_gate_and_resume_keep_sampling_anchor_and_counters(self):
        r=self.runner();r.current_learning_iteration=600;r._freeze_inference()
        self.assertFalse(next(r.actor_critic.long_history.parameters()).requires_grad)
        r.env.gait_curriculum.commit(1);r._freeze_inference()
        self.assertTrue(next(r.actor_critic.long_history.parameters()).requires_grad)
        self.assertFalse(r.actor_critic.log_std.requires_grad)
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'model_600.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner();loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,600)
            self.assertEqual(loaded.env.gait_curriculum.stage,1)
            self.assertEqual(loaded.actor_critic.exploration_gain,.5)
            obs=r.env.source_anchor_observations
            torch.testing.assert_close(r.policy_anchor(obs),loaded.policy_anchor(obs))

    def test_bank_histories_match_recorded_state_and_phase(self):
        env,cfg,bank=fixture22(self.source)
        states=torch.tensor(bank['states']);history=torch.tensor(bank['histories'])
        self.assertEqual(tuple(history.shape),(64,66,98))
        self.assertTrue(torch.isfinite(history).all())
        scale=self.source['config']['normalization']['obs_scales']
        default=torch.tensor([self.source['config']['init_state']['default_joint_angles'][n] for n in bank['dof_names']])
        torch.testing.assert_close(history[:,-1,9:38],(states[:,13:42]-default)*scale['dof_pos'])
        torch.testing.assert_close(history[:,-1,38:67],states[:,42:]*scale['dof_vel'])
        gravity=quat_inverse_rotate(states[:,3:7],torch.tensor([0.,0.,-1.]).expand(64,-1))
        torch.testing.assert_close(history[:,-1,6:9],gravity,rtol=1e-5,atol=1e-6)
        angle=torch.tensor(bank['phases'])*2*torch.pi
        torch.testing.assert_close(history[:,-1,-2:],torch.stack((angle.sin(),angle.cos()),-1),atol=5e-7,rtol=1e-5)
        self.assertEqual(bank['source_model_sha256'],'4edb874efe478baea76af598db86e923c24d62e1653542bf144111a453971ac8')

    def test_actual_reset_handles_empty_and_full_bank_branch_and_history(self):
        fake,cfg,bank=fixture22(self.source)
        class Base:
            def compute_observations(self):
                self.obs_history.append(torch.zeros(4,98));self.obs_history.pop(0)
                self.obs_buf=torch.cat(self.obs_history,-1)
        class Reset:
            @staticmethod
            def reset_idx(env,ids):
                env.dof_vel[ids]=env._reset_reference[:,42:]
                env.captured=env._reset_reference.clone()
        ns=dict(torch=torch,X1AMPExp21Env=Base,GaitCurriculum=GaitCurriculum,LeggedRobot=Reset)
        definitions('humanoid/envs/x1/x1_amp_exp22_env.py',ns)
        env=object.__new__(ns['X1AMPExp22Env'])
        env.cfg=configuration21()['X1AMPExp21Cfg']()
        env.device='cpu';env.gait_curriculum=ns['PolicyStateCurriculum']()
        env.bank={k:torch.tensor(bank[k]) for k in ('states','phases','histories','amp_states')}
        env.reference_guide=type('Ref',(),dict(sole_height=lambda _,s:torch.zeros(len(s),2),nearest_phase=lambda _,q,dq:torch.zeros(len(q))))()
        env.motion_library=fake.motion_library
        env.standing_state=torch.tensor(self.source['metadata']['control']['standing_state'])
        env.hard_dof_limits=torch.stack((fake.motion_library.fk.lower,fake.motion_library.fk.upper),-1)
        env._draw_commands=lambda n:torch.zeros(n,3)
        env.standing_commands=SmoothCommands(4,'cpu',75,.02)
        env.reset_standing=torch.ones(4,dtype=torch.bool);env.commands=torch.zeros(4,3)
        env.obs_history=[torch.zeros(4,98) for _ in range(66)]
        env.num_envs=4;env.privileged_obs_buf=torch.zeros(4,101)
        env.amp_history=torch.zeros(4,3,86);env.amp_valid_history=torch.zeros(4,3,dtype=torch.bool)
        env.actions=torch.zeros(4,29);env.last_actions=torch.zeros(4,29)
        env.last_last_actions=torch.zeros(4,29);env.last_dof_vel=torch.zeros(4,29);env.dof_vel=torch.zeros(4,29)
        for name in ('zero_steps','stop_hold_steps','previous_moving','stop_pending','stand_gate','gait_phase','completed_cycles'):setattr(env,name,torch.zeros(4))
        ids=torch.arange(4)
        for mode in ('rsi','standing'):
            env.evaluation_reset_mode=mode;env.reset()
            self.assertIsNone(env._bank_pending)
            self.assertTrue(torch.isfinite(env.amp_history).all())
            if mode=='rsi':
                self.assertFalse(env.reset_standing.any())
                torch.testing.assert_close(env.obs_buf[:,-31:-2],env.actions)
                torch.testing.assert_close(env.obs_buf[:,-89:-60],env.captured[:,13:42]-torch.tensor([self.source['config']['init_state']['default_joint_angles'][n] for n in bank['dof_names']]))
                self.assertTrue((env.commands[:,0]>.59).all())
            else:self.assertTrue(env.reset_standing.all())

if __name__=='__main__':unittest.main()
