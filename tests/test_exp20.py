"""Native cycle, command timing, reward ordering and conditional AMP regression."""
import unittest
import copy
from types import SimpleNamespace

import torch

from humanoid.amp.exp20_commands import SmoothCommands
from humanoid.amp.exp20_rewards import similarity,raw_violation,margin_cost,support_terms
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.motion import MotionLibrary
from test_amp import MOTIONS,URDF,KEYS
from humanoid.amp.runner import AMPOnPolicyRunner
from test_exp117 import configuration,definitions,grounded_library
from test_exp110 import dh_config,DHFakeEnv


def reward_environment(lib,reference,count=4):
    """Run the actual reward inheritance chain on synthetic physical tensors."""
    import humanoid.amp.gait_rewards as gait
    from humanoid.amp.motion import amp_features,quat_mul,quat_rotate,quat_inverse_rotate
    from humanoid.amp.control import sync_policy_commands
    ns = dict(torch=torch,BaseTask=object,LeggedRobotCfg=object,amp_features=amp_features,quat_mul=quat_mul,
        quat_rotate=quat_rotate,quat_inverse_rotate=quat_inverse_rotate,similarity=similarity,
        raw_violation=raw_violation,margin_cost=margin_cost,support_terms=support_terms,
        sync_policy_commands=sync_policy_commands)
    for name in ('moving_speed_quality','tracking_denominator','style_multiplier','scuff_cost',
        'sole_heights','lowest_sole_kinematics','backward_progress','backward_velocity_reward','braking_leg_gate','motion_velocity_reward'):
        ns[name] = getattr(gait,name)
    definitions('humanoid/envs/base/legged_robot.py',ns)
    definitions('humanoid/envs/x1/x1_amp_env.py',ns)
    for suffix in ('11','3','14','15','16','17','18','112','113','20'):
        definitions('humanoid/envs/x1/x1_amp_exp'+suffix+'_env.py',ns)
    cls = ns['X1AMPExp20Env']
    env = object.__new__(cls)
    env.cfg = configuration20()['X1AMPExp20Cfg']()
    env.cfg.env.capture_replay_state = False
    env.device,env.dt,env.num_envs = 'cpu',.02,count
    env.motion_library,env.reference_guide = lib,reference
    env.dof_names = lib.dof_names
    env.pitch_indices = reference.pitch_indices
    env.leg_indices = [i for i,n in enumerate(lib.dof_names) if any(s in n for s in ('hip_','knee_','ankle_'))]
    env.posture_groups = {
        'waist':[i for i,n in enumerate(lib.dof_names) if n.startswith('lumbar_')],
        'upper':[i for i,n in enumerate(lib.dof_names) if any(s in n for s in ('shoulder','elbow','wrist'))],
        'leg_lateral':[i for i,n in enumerate(lib.dof_names) if any(s in n for s in ('hip_roll','hip_yaw','ankle_roll'))]}
    env.waist_indices,env.upper_indices,env.lateral_indices = [env.posture_groups[k] for k in ('waist','upper','leg_lateral')]
    env.native_foot_ids = [0,1]
    env.gait_phase = torch.linspace(.1,.8,count)
    state = reference.state(env.gait_phase)
    env.root_states = state[:,:13].clone()
    env.base_quat = env.root_states[:,3:7]
    env.base_lin_vel = quat_inverse_rotate(env.base_quat,env.root_states[:,7:10])
    env.base_ang_vel = quat_inverse_rotate(env.base_quat,env.root_states[:,10:13])
    env.projected_gravity = quat_inverse_rotate(env.base_quat,torch.tensor([[0.,0.,-1.]]).expand(count,-1))
    env.env_origins = torch.zeros(count,3)
    env.dof_pos,env.dof_vel = state[:,13:42].clone(),state[:,42:].clone()
    env.standing_state = lib.state(torch.tensor([lib.names.index('F1_B1_stand_to_walk')]),torch.tensor([.22]))[0]
    env.standing_gravity = quat_inverse_rotate(env.standing_state[3:7],torch.tensor([0.,0.,-1.]))
    env.default_dof_pos = env.standing_state[13:42][None]
    env.hard_dof_limits = torch.stack((lib.fk.lower,lib.fk.upper),-1)
    env.dof_pos_limits = env.hard_dof_limits.clone()
    env.feet_indices,env.key_body_indices = torch.tensor([0,1]),torch.arange(6)
    local,rot = lib.fk.key_transforms(env.dof_pos)
    env.rigid_state = torch.zeros(count,6,13)
    env.rigid_state[:,:,:3] = env.root_states[:,None,:3]+quat_rotate(env.base_quat[:,None].expand(-1,6,-1),local)
    env.rigid_state[:,:,3:7] = quat_mul(env.base_quat[:,None].expand(-1,6,-1),rot)
    env.contact_forces = torch.zeros(count,6,3)
    height = reference.sole_height(state)
    env.contact_forces[:,:2,2] = (height<.008).float()*360
    env.penalised_contact_indices = torch.tensor([2,3,4,5])
    env.foot_com_local = torch.zeros(2,3)
    env.sole_vertices = [v for _,v in reference.feet]
    env.standing_commands = SmoothCommands(count,'cpu',75,.02)
    env.standing_commands.value[:] = env.cfg.commands.native_speed
    env.standing_commands.active_goal[:] = env.cfg.commands.native_speed
    env.standing_commands.requested[:,0] = env.cfg.commands.native_speed
    env.commands = env.standing_commands.applied()
    env.command_ranges = dict(lin_vel_x=[env.cfg.commands.native_speed]*2,lin_vel_y=[0.,0.],ang_vel_yaw=[0.,0.])
    env.raw_violation_sum = torch.zeros(count,29)
    env.actions,env.last_actions,env.last_dof_vel,env.torques = [torch.zeros(count,29) for _ in range(4)]
    env.target_clip_counts,env.torque_clip_counts = torch.zeros(count,29),torch.zeros(count,29)
    env.pd_target_raw,env.pd_target_effective = env.dof_pos.clone(),env.dof_pos.clone()
    env.reset_buf,env.time_out_buf,env.reset_standing = [torch.zeros(count,dtype=torch.bool) for _ in range(3)]
    env.episode_length_buf = torch.ones(count,dtype=torch.long)*200
    env.zero_steps,env.stop_hold_steps = torch.zeros(count,dtype=torch.long),torch.zeros(count,dtype=torch.long)
    env.previous_moving,env.stop_pending = torch.ones(count,dtype=torch.bool),torch.zeros(count,dtype=torch.bool)
    env.stand_gate = torch.zeros(count)
    env.amp_history = torch.zeros(count,3,86)
    env.amp_valid_history = torch.ones(count,3,dtype=torch.bool)
    env.obs_scales = env.cfg.normalization.obs_scales
    env.commands_scale = torch.tensor([env.obs_scales.lin_vel,env.obs_scales.lin_vel,env.obs_scales.ang_vel])
    env.total_mass = torch.ones(count,1)*36.46
    env.rew_buf = torch.zeros(count)
    env.extras = {}
    env._iteration = 800
    env.reward_scales = {name:getattr(env.cfg.rewards.scales,name) for name in dir(env.cfg.rewards.scales) if not name.startswith('_')}
    env._prepare_reward_function()
    return env


def configuration20():
    ns = configuration()
    definitions('humanoid/envs/x1/x1_amp_exp20_config.py',ns)
    return ns


class Exp20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        cls.lib = grounded_library()
        cls.reference = NativeCycle(cls.lib)

    def test_native_state_derivatives_history_and_margin(self):
        r = self.reference
        p = torch.tensor([-.0001,.2,.5,.9999,1.0001])
        epsilon = .0005
        s = r.state(p)
        delta = (r.state(p+epsilon/r.period)-r.state(p-epsilon/r.period))/(2*epsilon)
        torch.testing.assert_close(delta[:,13:42],s[:,42:],atol=.008,rtol=.01)
        torch.testing.assert_close(delta[:,:3],s[:,7:10],atol=.003,rtol=.01)
        torch.testing.assert_close(r.state(p+1)[:,:3]-s[:,:3],r.displacement.expand(len(p),-1),atol=1e-5,rtol=1e-5)
        self.assertEqual(tuple(r.history(p).shape),(5,3,86))
        self.assertAlmostEqual(r.period,1.26)
        self.assertAlmostEqual(r.nominal_speed,.59537,places=4)
        report = r.audit(630)
        self.assertFalse(report['passed'],'Known raw reference slip must not silently pass')
        self.assertLess(report['joint_margin_max_violation_rad'],1e-5)

    def test_smooth_goals_no_repeat_restarts_and_auto_reset_hold(self):
        c = SmoothCommands(2,'cpu',75,.02)
        c.reset(torch.arange(2),torch.ones(2,dtype=torch.bool))
        c.requested[0,0] = c.speed
        for _ in range(74):
            c.advance(torch.zeros(2,dtype=torch.bool))
            self.assertEqual(float(c.applied()[0,0]),0)
        c.advance(torch.zeros(2,dtype=torch.bool))
        self.assertTrue(c.started[0] and c.transition[0])
        previous = 0.
        starts = 0
        for _ in range(100):
            # Playback writes the same goal each frame; this must not restart.
            c.requested[0,0] = c.speed
            c.advance(torch.zeros(2,dtype=torch.bool))
            starts += int(c.started[0])
            self.assertGreaterEqual(float(c.value[0]),previous-1e-7)
            previous = float(c.value[0])
        self.assertEqual(starts,0)
        self.assertFalse(c.transition.any())
        self.assertAlmostEqual(float(c.value[0]),c.speed,places=6)
        c.requested[0,0] = 0
        for _ in range(101): c.advance(torch.zeros(2,dtype=torch.bool))
        self.assertFalse(c.transition.any())
        self.assertEqual(float(c.value[0]),0)
        c.reset(torch.tensor([0]),torch.tensor([True]))
        c.advance(torch.tensor([True,False]))
        self.assertEqual(int(c.remaining[0]),75)

    def test_correction_names_and_sole_support_cache_are_explicit(self):
        r = self.reference
        with self.assertRaisesRegex(ValueError,'named joint order'):
            NativeCycle(self.lib,contact_correction=torch.zeros_like(r.q))
        payload = dict(dof_names=self.lib.dof_names,joint_correction=(torch.sin(torch.arange(64)[:,None]*torch.pi/63)*torch.arange(29)[None]*.0001).tolist())
        forward = NativeCycle(self.lib,contact_correction=payload)
        reverse_lib = MotionLibrary(MOTIONS,URDF,list(reversed(self.lib.dof_names)),KEYS,ground_clearance=.002)
        reverse = NativeCycle(reverse_lib,contact_correction=payload)
        phases = torch.linspace(.01,.99,32)
        torch.testing.assert_close(forward.state(phases)[:,13:42],reverse.state(phases)[:,13:42].flip(-1))
        torch.manual_seed(5)
        normals = torch.randn(32,3)
        for (_,full),(_,support) in zip(self.lib.grounding.feet,r.feet):
            torch.testing.assert_close((normals@full.T).min(-1).values,(normals@support.T).min(-1).values,atol=1e-6,rtol=1e-6)

    def test_continuous_scores_recover_bad_pose_and_uncapped_target(self):
        target = torch.zeros(3,29)
        actual = torch.stack((torch.ones(29)*1.,torch.ones(29)*.5,torch.zeros(29)))
        quality = similarity(actual,target,.15)
        self.assertTrue((quality[1:]>quality[:-1]).all())
        raw = torch.ones(3,29)*2
        raw[1,0] = 1.9
        raw[2] = .9
        cost = raw_violation(raw,torch.zeros(29),torch.ones(29)).mean(-1)
        self.assertGreater(float(cost[0]),float(cost[1]))
        self.assertGreater(float(cost[1]),float(cost[2]))
        self.assertEqual(float(cost[2]),0)
        self.assertGreater(float(margin_cost(torch.ones(1,29),torch.zeros(29),torch.ones(29))),0)

    def test_swing_requires_clearance_unloading_and_opposite_support(self):
        target = torch.tensor([[.002,.03]]).expand(4,-1)
        height = torch.tensor([[.002,.03],[.002,.002],[.002,.03],[.002,.03]])
        force = torch.tensor([[360.,0.],[180.,180.],[0.,0.],[0.,360.]])
        swing,support,slip = support_terms(height,target,force,torch.ones(4)*360,torch.zeros(4,2))
        self.assertGreater(float(swing[0]),.99)
        self.assertEqual(float(swing[1]),0)
        self.assertEqual(float(swing[2]),0)
        self.assertEqual(float(swing[3]),0)
        self.assertEqual(float(support[2]),0)
        self.assertTrue((slip==0).all())

    def test_conditional_replay_rejects_transitions_and_pairs_categories(self):
        env = DHFakeEnv(self.lib)
        cfg = dh_config()
        cfg['discriminator'].update(conditional_experts=True,batch_size=8,num_updates=1,replay_size=8)
        runner = AMPOnPolicyRunner(env,cfg)
        seen = []
        def expert(moving):
            seen.append(moving.clone())
            return self.lib.sample(len(moving))
        env.sample_amp_expert_by_category = expert
        amp = self.lib.sample(12)
        valid = torch.tensor([True,False]*6)
        moving = torch.arange(12)%3==0
        runner.update_discriminator(amp,valid,moving)
        self.assertEqual(runner.replay.size,6)
        torch.testing.assert_close(runner.replay.data[:6],amp[valid])
        torch.testing.assert_close(runner.replay_categories[:6],moving[valid])
        self.assertEqual(len(seen),1)
        # Wrap both buffers and retain their exact row/category correspondence.
        second = self.lib.sample(6)
        labels = torch.tensor([True,True,False,False,True,False])
        runner.update_discriminator(second,torch.ones(6,dtype=torch.bool),labels)
        torch.testing.assert_close(runner.replay.data[[6,7,0,1,2,3]],second)
        torch.testing.assert_close(runner.replay_categories[[6,7,0,1,2,3]],labels)

    def test_exp20_configuration_keeps_inference_and_pd_contract(self):
        ns = configuration20()
        source,target = ns['X1AMPExp117Cfg'],ns['X1AMPExp20Cfg']
        for section in ('env','control','asset','sim','noise','normalization','domain_rand'):
            self.assertIs(getattr(source,section),getattr(target,section))
        self.assertFalse(target.amp.reference_state_initialization)
        self.assertTrue(target.amp.require_reference_acceptance)
        cfg = ns['X1AMPExp20CfgPPO']()
        self.assertEqual((cfg.runner.num_steps_per_env,cfg.algorithm.num_learning_epochs,cfg.algorithm.num_mini_batches),(24,2,4))
        self.assertEqual((cfg.runner.max_iterations,cfg.runner.policy_transfer_source_iteration),(3000,7000))
        self.assertEqual(cfg.runner.warm_start_mode,'policy_only')
        self.assertEqual(cfg.discriminator.num_updates,1)

    def test_actual_reward_chain_terminal_snapshot_and_style_masks(self):
        env = reward_environment(self.lib,self.reference)
        env.standing_commands.transition[1] = True
        env.reset_buf[2] = True
        env.compute_reward()
        self.assertTrue(torch.isfinite(env.rew_buf).all())
        self.assertFalse(env.extras['amp_valid'][1])
        self.assertEqual(float(env.extras['amp_style_multiplier'][1]),0)
        self.assertAlmostEqual(float(env.extras['amp_style_multiplier'][0]),.5)
        self.assertLess(float(env.rew_buf[2]),-5)
        snapshot = env.extras['amp_obs'].clone()
        env.amp_history.zero_()
        torch.testing.assert_close(env.extras['amp_obs'],snapshot)
        self.assertIn('Reference/upper_rmse_rad',env.extras['training_metrics'])

    def test_stop_retains_phase_and_command_request_does_not_jump_applied(self):
        env = reward_environment(self.lib,self.reference)
        env.obs_history = [torch.zeros(4,98)]
        env.obs_buf,env.privileged_obs_buf = torch.zeros(4,6468),torch.zeros(4,101)
        previous = env.gait_phase.clone()
        env.standing_commands.started[:] = True
        env.standing_commands.start[:] = env.cfg.commands.native_speed
        env.standing_commands.active_goal[:] = 0
        env._sync_phase()
        torch.testing.assert_close(previous,env.gait_phase)
        current = env.commands.clone()
        env.set_evaluation_command(0)
        torch.testing.assert_close(current,env.commands)
        for _ in range(5): env.set_evaluation_command(0)
        torch.testing.assert_close(current,env.commands)


if __name__=='__main__': unittest.main()
