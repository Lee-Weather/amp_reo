"""Real950 continuation and staged low-AMP entry contracts."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp23_curriculum import Exp23Curriculum
from humanoid.amp.exp23_runner import Exp23AMPOnPolicyRunner
from test_exp22 import fixture22,configuration21,definitions,dictionary

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP23_SOURCE_MODEL',str(ROOT/'czy/data/exp2.2/model_950.pt')))

def fixture23(source):
    env,cfg,bank=fixture22(source)
    ns=configuration21();definitions('humanoid/envs/x1/x1_amp_exp22_config.py',ns)
    definitions('humanoid/envs/x1/x1_amp_exp23_config.py',ns)
    cfg.update(dictionary(ns['X1AMPExp23CfgPPO']()))
    env.gait_curriculum=Exp23Curriculum()
    env.checkpoint_training_metadata=lambda:dict(curriculum='relaxed_anchor_low_amp_v1',
        reference_source_sha256=source['metadata']['training']['reference_source_sha256'])
    return env,cfg

class Exp23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        r=Exp23AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_continuation_preserves_all_weights_adam_and_original_anchor(self):
        r=self.runner()
        for name,value in self.source['model_state_dict'].items():torch.testing.assert_close(r.actor_critic.state_dict()[name],value,rtol=0,atol=0)
        actual=r.optimizer.state_dict()['state'];expected=self.source['optimizer_state_dict']['state']
        self.assertEqual(actual.keys(),expected.keys())
        for key,entry in expected.items():
            for name,value in entry.items():torch.testing.assert_close(actual[key][name],value,rtol=0,atol=0)
        for name,value in self.source['infos']['exp21']['anchor_state'].items():torch.testing.assert_close(r.policy_anchor.state_dict()[name],value,rtol=0,atol=0)
        self.assertEqual(r.current_learning_iteration,0)
        self.assertEqual(r.training_origin['source_iteration'],950)
        self.assertEqual(r.env.gait_curriculum.last_iteration,0)
        self.assertIsNotNone(r.env.gait_curriculum.baseline)
        self.assertFalse(r.actor_critic.log_std.requires_grad)

    def test_soft_constraint_reduced_tenfold_without_changing_noise(self):
        r=self.runner();obs=r.env.source_anchor_observations
        weak=float(r.policy_regularization_loss(obs));self.assertGreater(weak,0)
        r.alg_cfg['source_anchor_coef']=10.
        strong=float(r.policy_regularization_loss(obs));self.assertAlmostEqual(strong,10*weak,places=6)
        self.assertEqual(r.actor_critic.exploration_gain,.5)
        self.assertEqual(r.alg_cfg['source_max_pd_rms'],.04)

    def test_actual_ppo_updates_actor_keeps_features_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict())
        metrics=r.update_policy(r.collect_rollout())
        self.assertGreater(metrics['policy_updates'],0)
        self.assertEqual(metrics['critic_only'],0.)
        self.assertEqual(metrics['source_guard_rejected'],0.)
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for name,value in before.items():
            if name.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(value,r.actor_critic.state_dict()[name],rtol=0,atol=0)
        r.current_learning_iteration=1
        r.env.gait_curriculum.add([[10,145,4,0],[10,145,4,0]],1000,500)
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'model_1.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,1)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())

class EntryTests(unittest.TestCase):
    def controller(self):
        c=Exp23Curriculum();c.baseline=dict(standing=dict(mean_seconds=14.1,non_timeout_rate=.67),rsi=dict(mean_seconds=11.66,non_timeout_rate=.77))
        return c

    def test_first200_paused_then_two_windows_enter_only_low_amp(self):
        c=self.controller()
        for i in (50,100,150,200):
            c.add([[200,200*14.5,80,0],[200,200*14.5,80,0]],10000,6000)
            report=c.finish(i)
            self.assertEqual(c.weights['amp'],0.)
            self.assertEqual(report['next_stage'],int(i>=200))
        c.commit(1);self.assertEqual(c.weights['amp'],.01)
        for i in (250,300):
            c.add([[200,200*12,150,0],[200,200*12,150,0]],10000,6000)
            report=c.finish(i)
        self.assertEqual(report['next_stage'],0)
        c.commit(0);self.assertEqual(c.weights['amp'],0.)

    def test_no_promotion_on_one_branch_or_baseline_regression(self):
        for means in ((15.,13.),(13.,15.)):
            c=self.controller()
            for i in (50,100,150,200):
                c.add([[200,200*means[0],80,0],[200,200*means[1],80,0]],10000,6000)
                report=c.finish(i)
            self.assertFalse(report['qualified']);self.assertEqual(report['next_stage'],0)
        c=self.controller();c.baseline['standing']['mean_seconds']=19.
        for i in (50,100,150,200):
            c.add([[200,200*17,80,0],[200,200*15,80,0]],10000,6000)
            report=c.finish(i)
        self.assertFalse(report['qualified'])
        clone=Exp23Curriculum();clone.load_state_dict(c.state_dict())
        self.assertEqual(clone.state_dict(),c.state_dict())

if __name__=='__main__':unittest.main()
