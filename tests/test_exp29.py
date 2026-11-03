"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP29_SOURCE_MODEL',str(ROOT/'czy/data/exp2.8/model_7500.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24,25,26,27,28,29):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp29CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp29Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='command_scaled_phase_low_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp25AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,7500)
        self.assertEqual(r.training_origin['new_iteration'],7500)
        for name,value in self.source['model_state_dict'].items():
            torch.testing.assert_close(r.actor_critic.state_dict()[name],value,rtol=0,atol=0)
        for k,entry in self.source['optimizer_state_dict']['state'].items():
            for name,value in entry.items():torch.testing.assert_close(r.optimizer.state_dict()['state'][k][name],value,rtol=0,atol=0)
        for k,v in self.source['discriminator_state_dict'].items():
            torch.testing.assert_close(r.discriminator.state_dict()[k],v,rtol=0,atol=0)
        for k,entry in self.source['disc_optimizer_state_dict']['state'].items():
            for name,v in entry.items():torch.testing.assert_close(r.disc_optimizer.state_dict()['state'][k][name],v,rtol=0,atol=0)
        for k,v in self.source['infos']['exp21']['anchor_state'].items():
            torch.testing.assert_close(r.policy_anchor.state_dict()[k],v,rtol=0,atol=0)
        self.assertEqual(r.env.gait_curriculum.window_start,7500)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window']['pooled100'])

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,7501)
        self.assertEqual(r.env.gait_curriculum.d_updates,7301)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],.01)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_7501.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,7501)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,7500)
            state=torch.load(Path(tmp)/'model_7500.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class PhaseTests(unittest.TestCase):
    def test_actual_reference_derivative_matches_transition_time_law(self):
        from test_exp117 import grounded_library
        from humanoid.amp.native_reference import NativeCycle
        lib=grounded_library();ref=NativeCycle(lib)
        standing=torch.load(SOURCE,map_location='cpu',weights_only=False)['metadata']['control']['standing_state']
        standing=torch.tensor(standing)
        t=torch.tensor([.25,.7,1.4,1.95]);duration=2.;eps=.0005
        def pose(t):
            u=t/duration;blend=3*u*u-2*u**3;derivative=6*u*(1-u)/duration
            phase=.17+(t**3/duration**2-.5*t**4/duration**3)/ref.period
            return ref.blended_pose(phase,standing,blend,derivative,phase_rate_scale=blend)
        actual=pose(t);finite=(pose(t+eps)[:,13:42]-pose(t-eps)[:,13:42])/(2*eps)
        torch.testing.assert_close(actual[:,42:],finite,atol=.004,rtol=.01)
        phase=torch.tensor([.15,.7]);ones=torch.ones(2);zero=torch.zeros(2)
        torch.testing.assert_close(ref.blended_pose(phase,standing,ones,zero),
            ref.blended_pose(phase,standing,ones,zero,phase_rate_scale=ones),rtol=0,atol=0)
        torch.testing.assert_close(ref.blended_pose(phase,standing,zero,zero,phase_rate_scale=zero)[:,42:],torch.zeros(2,29))

    def test_actual_env_and_replay_share_clock_and_preserve_quiet(self):
        from types import SimpleNamespace
        from humanoid.amp.exp29_phase import phase_speed_scale,advance_phase
        class Base:pass
        ns=dict(X1AMPExp28Env=Base,Exp29Curriculum=Exp29Curriculum,phase_speed_scale=phase_speed_scale,advance_phase=advance_phase)
        definitions('humanoid/envs/x1/x1_amp_exp29_env.py',ns)
        env=object.__new__(ns['X1AMPExp29Env']);env.gait_phase=torch.tensor([.2,.2,.2])
        env.standing_commands=SimpleNamespace(value=torch.tensor([0.,.3,.6]))
        env.reference_guide=SimpleNamespace(period=1.26);env.dt=.02
        env.cfg=SimpleNamespace(commands=SimpleNamespace(native_speed=.6))
        env._advance_reference_phase(torch.ones(3,dtype=torch.bool))
        torch.testing.assert_close(env.gait_phase,advance_phase(torch.tensor([.2,.2,.2]),torch.tensor([0.,.3,.6]),.02,1.26,.6))
        self.assertEqual(float(env.gait_phase[0]),float(torch.tensor(.2)))
        self.assertAlmostEqual(float(env.gait_phase[1]-.2),.01/1.26,places=6)
        # Sim2sim production entry selects the shared clock only from checkpoint metadata.
        script=(ROOT/'scripts/amp_tools/sim2sim_amp_native.py').read_text()
        self.assertIn("meta['training'].get('phase_clock')=='command_scaled_v1'",script)
        self.assertIn('advance_phase(phase,applied,dt,ref.period',script)

    def test_config_and_all_strengths_unchanged(self):
        ns=configuration21()
        for n in (22,23,24,25,26,27,28,29):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        self.assertEqual(dictionary(ns['X1AMPExp28Cfg']()),dictionary(ns['X1AMPExp29Cfg']()))
        c=Exp29Curriculum();c.last_iteration=7500
        self.assertEqual((c.forward_multiplier,c.balance_multiplier,c.swing_multiplier,c.weights['amp']),(1.,1.,2.,.01))

    def test_mujoco_contact_force_world_direction_and_load(self):
        import mujoco
        import numpy as np
        from mujoco_diagnostics import foot_contact_forces
        m=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.002"/><worldbody><geom name="floor" type="plane" size="2 2 .1"/><body pos="0 0 .1"><freejoint/><geom name="foot" type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m)
        for _ in range(1000):mujoco.mj_step(m,d)
        force,flags=foot_contact_forces(m,d,[[1],[0]])
        self.assertEqual(flags.tolist(),[1,1]);self.assertAlmostEqual(force[0,2],9.81,places=3)
        np.testing.assert_allclose(force[0],-force[1],atol=1e-8)

if __name__=='__main__':unittest.main()
