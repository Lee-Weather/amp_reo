"""Directional incentives, real nonquiet RSI support, and COM-aware rolling kinematics."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import torch

from humanoid.amp.gait_rewards import backward_progress, lowest_sole_kinematics
from humanoid.amp.motion import MotionLibrary, quat_inverse_rotate
from humanoid.amp.stationary import CommandMotionSampler, sample_directional_commands
from test_amp import MOTIONS, URDF, JOINTS, KEYS
from test_exp3 import definitions


class Exp17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rolling_point_uses_com_arm_and_retains_translation(self):
        state=torch.zeros(2,2,13);state[:,:,6]=1;state[:,:,2]=.05
        # Selected point=(.1,0,-.05), COM=(.02,.03,0), omega=(0,1,0).
        # v_COM=(.05,0,.08) cancels omega cross (point-COM) exactly.
        state[:,:,7]=.05;state[:,:,9]=.08;state[:,:,11]=1
        mesh=torch.tensor([[.1,0.,-.05],[-.1,0.,.05]])
        com=torch.tensor([[.02,.03,0.],[.02,.03,0.]])
        state[1,:,7]+=.2
        height,speed=lowest_sole_kinematics(state,[mesh,mesh],com,torch.zeros(2))
        torch.testing.assert_close(height,torch.zeros(2,2))
        torch.testing.assert_close(speed,torch.tensor([[0.,0.],[.2,.2]]),atol=1e-7,rtol=0)

    def test_backward_bonus_does_not_reward_rest_wrong_direction_or_falls(self):
        cmd=torch.tensor([-.2,-.2,-.2,-.2,0.,.1])
        v=torch.tensor([0.,.1,-.1,-1.,-1.,-1.])
        torch.testing.assert_close(backward_progress(cmd,v),torch.tensor([0.,0.,.5,1.,0.,0.]))
        ns=dict(torch=torch,backward_progress=backward_progress,X1AMPExp16Env=object)
        definitions('humanoid/envs/x1/x1_amp_exp17_env.py',ns)
        env=ns['X1AMPExp17Env']();env.commands=torch.tensor([[-.2,0,0.]]*4);env.base_lin_vel=torch.tensor([[-.2,0,0.]]*4)
        env.root_states=torch.zeros(4,13);env.root_states[:,2]=.6;env.root_states[1,2]=.2
        env.env_origins=torch.zeros(4,3);env.projected_gravity=torch.tensor([[0.,0.,-1.]]*4);env.projected_gravity[2,2]=-.4
        env.reset_buf=torch.tensor([0,0,0,1]);env.time_out_buf=torch.zeros(4,dtype=torch.bool)
        torch.testing.assert_close(env._reward_backward_progress(),torch.tensor([1.,0.,0.,0.]))

    def test_direction_exposure_preserves_full_range_and_fixed_commands(self):
        torch.manual_seed(5)
        ranges={'lin_vel_x':[-.2,.65],'lin_vel_y':[0.,0.],'ang_vel_yaw':[0.,0.]}
        c=sample_directional_commands(ranges,50000,'cpu')
        self.assertLess(abs(float((c[:,0]==0).float().mean())-.2),.01)
        self.assertLess(abs(float((c[:,0]<0).float().mean())-.3),.01)
        self.assertTrue((c[:,1:]==0).all());self.assertLess(float(c[:,0].min()),-.199);self.assertGreater(float(c[:,0].max()),.649)
        for v in [0.,-.2,.1,.65]:
            ranges['lin_vel_x']=[v,v]
            torch.testing.assert_close(sample_directional_commands(ranges,100,'cpu')[:,0],torch.full((100,),v))

    def test_full_real_command_grid_has_nonquiet_signed_rsi_support(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lib=MotionLibrary(MOTIONS,URDF,JOINTS,KEYS,ground_clearance=.002,
                weights={'F1_B1_stand_to_walk':1.,'F1_B2_walk_to_stand':1.,'F1_B3_continuous_walk':1.,'F1_B4_walk_backwards':2.})
        sampler=CommandMotionSampler(lib,exclude_quiet_for_moving=True,relative_speed_tolerance=.5)
        cmd=torch.zeros(2177,3);cmd[128:,0]=torch.linspace(-.2,.65,2049)
        ids,times=sampler.sample_matching(cmd);state=lib.state(ids,times)
        vx=quat_inverse_rotate(state[:,3:7],state[:,7:10])[:,0]
        tol=(.025+.5*cmd[:,0].abs()).clamp(max=.15)
        moving=cmd[:,0].abs()>.05
        self.assertTrue(((vx-cmd[:,0]).abs()[128:] <= tol[128:]+1e-6).all())
        self.assertTrue((vx[moving]*cmd[moving,0]>0).all())
        pool_keys=sampler.ids*100000+(sampler.times*lib.fps[sampler.ids]).round().long()
        keys=ids*100000+(times*lib.fps[ids]).round().long();selected=torch.searchsorted(pool_keys,keys)
        self.assertTrue((pool_keys[selected]==keys).all());self.assertFalse(sampler.quiet[selected[moving]].any())
        self.assertTrue(sampler.quiet[selected[:128]].all())

    def test_lateral_tracking_is_not_relaxed_at_high_forward_speed(self):
        ns=dict(torch=torch,X1AMPExp16Env=object)
        definitions('humanoid/envs/x1/x1_amp_exp17_env.py',ns)
        env=ns['X1AMPExp17Env']();env.commands=torch.tensor([[.6,0.,0.]]*3)
        env.base_lin_vel=torch.tensor([[.6,0.,0.],[.6,.1,0.],[.6,.2,0.]])
        env.cfg=SimpleNamespace(rewards=SimpleNamespace(tracking_lateral_std=.1))
        env._tracking_denominator=lambda:torch.full((3,),.3**2)
        torch.testing.assert_close(env._reward_tracking_lin_vel(),torch.exp(-torch.tensor([0.,1.,4.])))


if __name__=='__main__':unittest.main()
