"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp25_curriculum import Exp25Curriculum
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP25_SOURCE_MODEL',str(ROOT/'czy/data/exp2.4/model_1100.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24,25):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp25CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp25Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='pooled100_low_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp25AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,1100)
        self.assertEqual(r.training_origin['new_iteration'],1100)
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
        self.assertEqual(r.env.gait_curriculum.window_start,1100)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window'])

    def test_ramp_reaches_rollout_style_rewards(self):
        r=self.runner();step=r.env.step
        def wrapped(actions):
            obs,critic,reward,done,info=step(actions)
            info['amp_style_multiplier']=torch.full((4,),r.env.gait_curriculum.weights['amp'])
            return obs,critic,reward,done,info
        r.env.step=wrapped
        self.assertEqual(float(r.collect_rollout()['style_rewards'].abs().max()),0)
        r.env.gait_curriculum.commit(1);r.env.gait_curriculum.last_iteration+=50
        batch=r.collect_rollout()
        self.assertGreater(float(batch['style_rewards'].mean()),0)
        self.assertAlmostEqual(r.env.gait_curriculum.weights['amp'],.005)

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,1101)
        self.assertEqual(r.env.gait_curriculum.d_updates,901)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],0)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_1101.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,1101)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,1100)
            state=torch.load(Path(tmp)/'model_1100.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class StageTests(unittest.TestCase):
    def controller(self):
        c=Exp25Curriculum();c.last_iteration=c.window_start=1100;c.run_start_iteration=1100
        c.d_updates=900
        c.baseline=dict(standing=dict(mean_seconds=16.35,non_timeout_rate=.439),rsi=dict(mean_seconds=15.415,non_timeout_rate=.488))
        return c

    def window(self,c,i,seconds=16.,fail=.51):
        c.add([[200,200*seconds,200*fail,0]]*2,10000,6000)
        return c.finish(i)

    def test_pooled_entry_ramp_and_resume(self):
        c=self.controller()
        self.assertEqual(self.window(c,1150,16,.49)['next_stage'],0)
        r=self.window(c,1200,16,.48)
        self.assertTrue(r['qualified']);self.assertEqual(r['next_stage'],1)
        c.commit(1);self.assertEqual(c.weights['amp'],0)
        self.window(c,1250);self.assertAlmostEqual(c.weights['amp'],.005)
        self.window(c,1300);self.assertAlmostEqual(c.weights['amp'],.01)
        for i in (1350,1400,1450):self.window(c,i)
        self.assertEqual(c.stage,1);self.assertIsNone(c.stop_reason)
        clone=Exp25Curriculum();clone.load_state_dict(c.state_dict())
        self.assertEqual(c.state_dict(),clone.state_dict())

    def test_entry_does_not_use_old_windows_or_accept_baseline_regression(self):
        c=self.controller()
        self.window(c,1150,16,.51)
        r=self.window(c,1200,16,.59)
        self.assertFalse(r['qualified'])
        self.assertEqual(r['next_stage'],0)
        self.assertEqual(r['pooled100']['standing']['n'],400)
        # standing baseline+5pp is stricter than the global55% cap.
        c=self.controller()
        self.window(c,1150,16,.51);self.assertFalse(self.window(c,1200,16,.51)['qualified'])

    def test_no_old_stagnation_stop_and_hysteresis_after_entry(self):
        c=self.controller()
        for i in range(1150,1751,50):self.window(c,i,15,.58)
        self.assertIsNone(c.stop_reason)
        self.window(c,1800,16,.45);self.window(c,1850,16,.45)
        self.assertEqual(self.window(c,1900,16,.45)['next_stage'],1)
        c.commit(1)
        for i in (1950,2000):self.window(c,i,14,.58)
        self.assertIsNone(c.stop_reason)
        for i in (2050,2100):self.window(c,i,12,.65)
        self.assertIsNone(c.stop_reason)
        for i in (2150,2200):self.window(c,i,12,.65)
        self.assertIn('after_amp',c.stop_reason)

    def test_source_stop_flag_cleared_and_discriminator_counter_retained(self):
        from humanoid.amp.exp24_curriculum import Exp24Curriculum
        source=Exp24Curriculum();source.window_start=source.last_iteration=1100
        source.d_updates=900;source.stop_reason='old_stagnation';source.last_window=self.controller().baseline
        c=Exp25Curriculum();c.migrate(source.state_dict(),1100)
        self.assertEqual(c.d_updates,900);self.assertIsNone(c.stop_reason)
        self.assertEqual(c.run_start_iteration,1100)

if __name__=='__main__':unittest.main()
