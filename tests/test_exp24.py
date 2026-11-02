"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp24_curriculum import Exp24Curriculum
from humanoid.amp.exp24_runner import Exp24AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP24_SOURCE_MODEL',str(ROOT/'czy/data/exp2.3/model_200.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp24CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp24Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='overnight_preheated_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp24AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,200)
        self.assertEqual(r.training_origin['new_iteration'],200)
        for name,value in self.source['model_state_dict'].items():
            torch.testing.assert_close(r.actor_critic.state_dict()[name],value,rtol=0,atol=0)
        for k,entry in self.source['optimizer_state_dict']['state'].items():
            for name,value in entry.items():torch.testing.assert_close(r.optimizer.state_dict()['state'][k][name],value,rtol=0,atol=0)
        for k,v in self.source['infos']['exp21']['anchor_state'].items():
            torch.testing.assert_close(r.policy_anchor.state_dict()[k],v,rtol=0,atol=0)
        self.assertEqual(r.env.gait_curriculum.window_start,200)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window'])

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,201)
        self.assertEqual(r.env.gait_curriculum.d_updates,1)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],0)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_201.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,201)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,200)
            state=torch.load(Path(tmp)/'model_200.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class StageTests(unittest.TestCase):
    def controller(self):
        c=Exp24Curriculum();c.last_iteration=c.window_start=200
        c.baseline=dict(standing=dict(mean_seconds=15.3,non_timeout_rate=.55),rsi=dict(mean_seconds=14.18,non_timeout_rate=.585))
        return c

    def window(self,c,i,seconds=17,fail=.3):
        c.add([[200,200*seconds,200*fail,0]]*2,10000,6000)
        return c.finish(i)

    def test_preheat_stage_hold_retention_and_downgrade(self):
        c=self.controller()
        self.assertEqual(self.window(c,250)['next_stage'],0)
        self.assertEqual(self.window(c,300)['next_stage'],0)
        c.d_updates=100
        self.assertEqual(self.window(c,350)['next_stage'],1)
        c.commit(1)
        for i in (400,450,500):self.assertEqual(self.window(c,i)['next_stage'],1)
        self.assertEqual(self.window(c,550)['next_stage'],2)
        c.commit(2)
        # Retains stage2 under the stage1 entry gate, without meeting stage3.
        for i in (600,650):self.assertEqual(self.window(c,i,17,.3)['next_stage'],2)
        self.window(c,700,12,.8)
        self.assertEqual(self.window(c,750,12,.8)['next_stage'],1)
        c.commit(1)
        clone=Exp24Curriculum();clone.load_state_dict(c.state_dict())
        self.assertEqual(clone.state_dict(),c.state_dict())

    def test_stagnation_requires500_updates_and_respects_progress(self):
        for improve in (False,True):
            c=self.controller()
            for j in range(10):
                report=self.window(c,250+50*j,14+(.4 if improve and j>=5 else 0),.6)
                if j<9:self.assertIsNone(c.stop_reason)
            self.assertEqual(c.stop_reason is None,improve)

    def test_severe_regression_requires_two_both_branch_windows(self):
        c=self.controller();self.window(c,250,8,.9)
        self.assertIsNone(c.stop_reason)
        self.window(c,300,8,.9)
        self.assertIn('below10s',c.stop_reason)

    def test_swing_multiplier_changes_only_at_stage2(self):
        class Base:
            def _reward_reference_swing(self):return torch.tensor([.4,.7])
        ns=dict(X1AMPExp23Env=Base,Exp24Curriculum=Exp24Curriculum)
        definitions('humanoid/envs/x1/x1_amp_exp24_env.py',ns)
        env=object.__new__(ns['X1AMPExp24Env']);env.gait_curriculum=self.controller()
        before=env._reward_reference_swing();env.gait_curriculum.stage=2
        torch.testing.assert_close(env._reward_reference_swing(),before*2)

if __name__=='__main__':unittest.main()
