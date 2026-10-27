"""Reject zero-net velocity reward loops and preserve quiet holds after braking."""
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.gait_rewards import backward_progress, backward_velocity_reward, braking_leg_gate, tracking_denominator
from test_exp3 import definitions, experiment_config


class Exp18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_signed_reward_peak_rest_overspeed_and_complete_command_range(self):
        command = torch.linspace(-.2,-.050001,501)
        rest = torch.zeros_like(command)
        optimum = backward_velocity_reward(command,command)
        torch.testing.assert_close(backward_velocity_reward(command,rest),rest)
        self.assertTrue((optimum>0).all())
        self.assertTrue((backward_velocity_reward(command,.1+rest)<0).all())
        self.assertTrue((backward_velocity_reward(command,2.5*command)<optimum).all())
        c=torch.tensor([-.05,0.,.1,.65]);v=torch.full_like(c,-.2)
        torch.testing.assert_close(backward_velocity_reward(c,v),torch.zeros_like(c))
        for invalid in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):backward_velocity_reward(c,v,invalid)

    def test_zero_net_return_strokes_cannot_collect_positive_bonus(self):
        command = torch.linspace(-.2,-.050001,101,dtype=torch.float64)[:,None]
        # Faster forward return cancels a longer backward phase exactly.
        velocity=torch.cat((command.expand(-1,7),(-7*command/3).expand(-1,3)),1)
        self.assertLess(float(velocity.mean(1).abs().max()),1e-14)
        self.assertTrue((backward_progress(command,velocity).mean(1)>.6).all())
        self.assertTrue((backward_velocity_reward(command,velocity).mean(1)<0).all())
        torch.manual_seed(5)
        v=torch.randn(101,1000,dtype=torch.float64)*.3
        v-=v.mean(1,keepdim=True)
        self.assertTrue((backward_velocity_reward(command,v).mean(1)<=1e-12).all())

    def test_core_reward_prefers_real_tracking_to_rest_and_return_loop(self):
        command=torch.tensor(-.2,dtype=torch.float64)
        rest=torch.zeros(10,dtype=torch.float64)
        loop=torch.tensor([-.2]*7+[.2*7/3]*3,dtype=torch.float64)
        sigma=tracking_denominator(command)
        def score(v,new):
            bonus=1.5*backward_velocity_reward(command,v) if new else backward_progress(command,v)
            return float((1.25*torch.exp(-(v-command).square()/sigma)+bonus).mean())
        self.assertGreater(score(loop,False),score(rest,False))
        self.assertLess(score(loop,True),score(rest,True))
        self.assertGreater(score(command.expand(10),True),score(rest,True))

    def test_gate_relaxes_only_pending_braking_then_restores_full_quiet_cost(self):
        v=torch.tensor([[.4,0.],[.02,0.],[0.,0.],[.4,0.]])
        w=torch.tensor([.2,.01,0.,.2]);pending=torch.tensor([True,True,True,False])
        gate=braking_leg_gate(v,w,pending)
        self.assertLess(float(gate[0]),1e-8);self.assertGreater(float(gate[1]),.9)
        torch.testing.assert_close(gate[2:],torch.ones(2))
        class Parent:
            def _reward_stand_dof_vel(self):return torch.full((4,),10.)
            def _reward_stand_pose(self):return torch.full((4,),7.)
        ns=dict(torch=torch,X1AMPExp17Env=Parent,backward_velocity_reward=backward_velocity_reward,braking_leg_gate=braking_leg_gate)
        definitions('humanoid/envs/x1/x1_amp_exp18_env.py',ns)
        env=ns['X1AMPExp18Env']();env.base_lin_vel=v;env.base_ang_vel=torch.zeros(4,3);env.base_ang_vel[:,2]=w
        env.stop_pending=pending;env.cfg=SimpleNamespace(rewards=SimpleNamespace(brake_leg_linear_scale=.08,brake_leg_angular_scale=.15))
        torch.testing.assert_close(env._reward_stand_dof_vel(),10*gate)
        torch.testing.assert_close(env._reward_stand_pose(),7*gate)
        env.stop_pending.zero_()
        torch.testing.assert_close(env._reward_stand_dof_vel(),torch.full((4,),10.))

    def test_fall_gate_never_awards_target_speed_and_controls_stay_inherited(self):
        ns=dict(torch=torch,X1AMPExp17Env=object,backward_velocity_reward=backward_velocity_reward,braking_leg_gate=braking_leg_gate)
        definitions('humanoid/envs/x1/x1_amp_exp18_env.py',ns)
        env=ns['X1AMPExp18Env']();env.commands=torch.tensor([[-.2,0.,0.]]*5);env.base_lin_vel=env.commands.clone()
        env.root_states=torch.zeros(5,13);env.root_states[:,2]=.6;env.root_states[1,2]=.2
        env.env_origins=torch.zeros(5,3);env.projected_gravity=torch.tensor([[0.,0.,-1.]]*5);env.projected_gravity[2,2]=-.4
        env.reset_buf=torch.tensor([0,0,0,1,1]);env.time_out_buf=torch.tensor([0,0,0,0,1],dtype=torch.bool)
        env.cfg=SimpleNamespace(rewards=SimpleNamespace(backward_speed_scale=.2))
        torch.testing.assert_close(env._reward_backward_progress(),torch.tensor([1.,0.,0.,0.,1.]))
        cfg=experiment_config()
        for n in (14,15,16,17,18):definitions('humanoid/envs/x1/x1_amp_exp%d_config.py'%n,cfg)
        a,b=cfg['X1AMPExp17Cfg'],cfg['X1AMPExp18Cfg']
        for key in ('control','env','asset','commands','amp','sim','noise','domain_rand'):
            self.assertIs(getattr(a,key),getattr(b,key))
        self.assertEqual(b.rewards.tracking_sigma,a.rewards.tracking_sigma)
        self.assertEqual(b.rewards.scales.feet_scuff,a.rewards.scales.feet_scuff)


if __name__=='__main__':unittest.main()
