"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp26_curriculum import Exp26Curriculum
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP26_SOURCE_MODEL',str(ROOT/'czy/data/exp2.5/model_3100.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24,25,26):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp26CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp26Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='dense_swing_low_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp25AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,3100)
        self.assertEqual(r.training_origin['new_iteration'],3100)
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
        self.assertEqual(r.env.gait_curriculum.window_start,3100)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window']['pooled100'])

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,3101)
        self.assertEqual(r.env.gait_curriculum.d_updates,2901)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],.01)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_3101.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,3101)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,3100)
            state=torch.load(Path(tmp)/'model_3100.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class SwingTests(unittest.TestCase):
    def test_environment_logs_and_reward_share_same_components_and_gate(self):
        from humanoid.amp.exp26_swing import dense_swing,ClearanceRuns
        from humanoid.amp.exp20_rewards import support_terms
        class Base:
            def _support_terms(self,h,t,f,w,s):return support_terms(h,t,f,w,s)
            def _reference_weight(self):return .1
            def _reference_valid(self):return self.valid
            def _reward_reference_swing(self):return self.swing_score*.1*self.valid
            def compute_reward(self):
                self.swing_score,_,_=self._support_terms(self.height,self.target,self.force,self.weight,self.speed)
                self.actual_reward=self._reward_reference_swing()*self.reward_scales['reference_swing']
                self.extras=dict(training_metrics={})
        ns=dict(X1AMPExp25Env=Base,dense_swing=dense_swing,ClearanceRuns=ClearanceRuns,Exp26Curriculum=Exp26Curriculum)
        definitions('humanoid/envs/x1/x1_amp_exp26_env.py',ns)
        env=object.__new__(ns['X1AMPExp26Env']);env.num_envs=2;env.dt=.02
        env.height=torch.tensor([[.004,0.],[.004,0.]])
        env.target=torch.tensor([[.035,0.],[.035,0.]])
        env.force=torch.full((2,2),250.);env.weight=torch.full((2,),500.);env.speed=torch.zeros(2,2)
        env.commands=torch.zeros(2,3);env.valid=torch.tensor([True,False])
        env.reward_scales={'reference_swing':.02};env.gait_curriculum=Exp26Curriculum()
        env.gait_curriculum.last_iteration=3300;env.swing_air_tracker=ClearanceRuns(2,'cpu',.02)
        env.compute_reward()
        self.assertGreater(float(env.actual_reward[0]),0);self.assertEqual(float(env.actual_reward[1]),0)
        metrics=env.extras['training_metrics']
        contributions=sum(float(metrics['Swing/'+side+'/'+term+'_reward'].sum())
            for side in ('left','right') for term in ('height','unload','other_support'))
        self.assertAlmostEqual(contributions,float(env.actual_reward.sum()),places=8)
        self.assertEqual(metrics['Swing/left/height_error_m'].numel(),1)

    def test_small_lift_below5mm_has_reward_and_keeps_original_support(self):
        from humanoid.amp.exp26_swing import dense_swing
        from humanoid.amp.exp20_rewards import support_terms
        from test_exp23 import definitions
        # Both feet loaded; the old clearance multiplier suppresses small lifts.
        height=torch.tensor([[0.,0.],[.004,0.]])
        target=torch.tensor([[.035,0.],[.035,0.]])
        force=torch.tensor([[250.,250.],[250.,250.]])
        weight=torch.tensor([500.,500.]);speed=torch.zeros(2,2)
        old=support_terms(height,target,force,weight,speed)
        new,detail=dense_swing(height,target,force,weight)
        self.assertEqual(float(old[0].sum()),0)
        self.assertGreater(float(new[0]),0);self.assertGreater(float(new[1]),float(new[0]))
        swapped,_=dense_swing(height.flip(-1),target.flip(-1),force.flip(-1),weight)
        torch.testing.assert_close(new,swapped)
        zero,_=dense_swing(height,target,torch.zeros_like(force),weight)
        self.assertEqual(float(zero.sum()),0)
        # Actual base hook and derived override: support/slip are unchanged.
        ns=dict(X1AMPExp113Env=object,support_terms=support_terms)
        definitions('humanoid/envs/x1/x1_amp_exp20_env.py',ns)
        class Base(ns['X1AMPExp20Env']):pass
        from humanoid.amp.exp26_swing import ClearanceRuns
        definitions('humanoid/envs/x1/x1_amp_exp26_env.py',dict_ns:=dict(ns,X1AMPExp25Env=Base,dense_swing=dense_swing,ClearanceRuns=ClearanceRuns,Exp26Curriculum=Exp26Curriculum))
        env=object.__new__(dict_ns['X1AMPExp26Env'])
        got=env._support_terms(height,target,force,weight,speed)
        torch.testing.assert_close(got[0],new)
        torch.testing.assert_close(got[1],old[1]);torch.testing.assert_close(got[2],old[2])

    def test_clearance_runs_count_once_after80ms_and_reset(self):
        from humanoid.amp.exp26_swing import ClearanceRuns
        tracker=ClearanceRuns(1,'cpu',.02);height=torch.tensor([[.011,.005]]);valid=torch.tensor([True])
        for _ in range(3):self.assertFalse(tracker.update(height,valid).any())
        self.assertEqual(tracker.update(height,valid).tolist(),[[True,False]])
        for _ in range(8):self.assertFalse(tracker.update(height,valid).any())
        tracker.reset(torch.tensor([0]));self.assertFalse(tracker.update(height,valid).any())
        tracker.update(height,torch.tensor([False]));self.assertEqual(float(tracker.seconds.sum()),0)

    def test_amp_stays_active_and_swing_ramps_and_relative_regression_stops(self):
        c=Exp26Curriculum();c.last_iteration=c.window_start=3100;c.d_updates=2900
        c.baseline=dict(standing=dict(mean_seconds=17.64,non_timeout_rate=.301),rsi=dict(mean_seconds=16.76,non_timeout_rate=.359))
        for i in range(3150,3301,50):
            c.add([[200,200*17,60,0]]*2,10000,6000);c.finish(i)
            self.assertEqual(c.weights['amp'],.01)
        self.assertEqual(c.swing_multiplier,2.);self.assertIsNone(c.stop_reason)
        for i in (3350,3400):
            c.add([[200,200*14,120,0]]*2,10000,6000);c.finish(i)
        self.assertIsNone(c.stop_reason)
        for i in (3450,3500):
            c.add([[200,200*14,120,0]]*2,10000,6000);c.finish(i)
        self.assertIn('source_regression',c.stop_reason)
        clone=Exp26Curriculum();clone.load_state_dict(c.state_dict());self.assertEqual(clone.state_dict(),c.state_dict())

if __name__=='__main__':unittest.main()
