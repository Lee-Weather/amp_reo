"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp28_curriculum import Exp28Curriculum
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP28_SOURCE_MODEL',str(ROOT/'czy/data/exp2.7/model_6300.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24,25,26,27,28):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp28CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp28Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='forward_correction_low_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp25AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,6300)
        self.assertEqual(r.training_origin['new_iteration'],6300)
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
        self.assertEqual(r.env.gait_curriculum.window_start,6300)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window']['pooled100'])

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,6301)
        self.assertEqual(r.env.gait_curriculum.d_updates,6101)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],.01)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_6301.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,6301)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,6300)
            state=torch.load(Path(tmp)/'model_6300.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class ForwardTests(unittest.TestCase):
    def test_actual_environment_cost_reward_logging_and_motion_gate(self):
        from humanoid.amp.exp28_tracking import forward_cost
        class Base:
            def compute_reward(self):
                self.actual = self._reward_corridor_forward()*self.reward_scales['corridor_forward']
                self.extras = {'training_metrics':{}}
        ns=dict(X1AMPExp27Env=Base,Exp28Curriculum=Exp28Curriculum,forward_cost=forward_cost)
        definitions('humanoid/envs/x1/x1_amp_exp28_env.py',ns)
        env=object.__new__(ns['X1AMPExp28Env'])
        env.base_lin_vel=torch.tensor([[.2,0.,0.],[1.,0.,0.],[1.,0.,0.]])
        env.commands=torch.tensor([[.6,0.,0.],[.6,0.,0.],[.6,0.,0.]])
        env.num_envs=3;env._reference_valid=lambda:torch.tensor([True,True,False])
        env.gait_curriculum=Exp28Curriculum();env.gait_curriculum.last_iteration=6400
        env.reward_scales={'corridor_forward':-.30*.02}
        env.compute_reward();metrics=env.extras['training_metrics']
        self.assertLess(float(env.actual[0]),0)
        torch.testing.assert_close(env.actual[0],env.actual[1])
        self.assertEqual(float(env.actual[2]),0)
        torch.testing.assert_close(metrics['Forward/reward'],env.actual[:2])
        self.assertEqual(float(metrics['Forward/multiplier'][0]),.5)
        loss=forward_cost(torch.tensor([.6, .8,1.,1.4]),torch.full((4,),.6))
        self.assertEqual(float(loss[0]),0);self.assertTrue(torch.all(loss[1:]>loss[:-1]))

    def test_only_forward_reward_changed_and_old_controls_kept(self):
        ns=configuration21()
        for n in (22,23,24,25,26,27,28):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        old=dictionary(ns['X1AMPExp27Cfg']());new=dictionary(ns['X1AMPExp28Cfg']())
        for k in old:
            if k!='rewards':self.assertEqual(old[k],new[k],k)
        a=old['rewards'];b=copy.deepcopy(new['rewards']);self.assertEqual(b['scales'].pop('corridor_forward'),-.30)
        self.assertEqual(a,b)

    def test_hold_old_costs_and_ramp_only_new_cost_after_migration(self):
        c=Exp28Curriculum();c.last_iteration=6300
        self.assertEqual(c.balance_multiplier,1);self.assertEqual(c.swing_multiplier,2)
        self.assertEqual(c.weights['amp'],.01);self.assertEqual(c.forward_multiplier,0)
        c.last_iteration=6400;self.assertEqual(c.forward_multiplier,.5)
        c.last_iteration=6500;self.assertEqual(c.forward_multiplier,1.)
        c.last_iteration=8300;self.assertEqual(c.forward_multiplier,1.)
        restored=Exp28Curriculum();restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.forward_multiplier,1.);self.assertEqual(restored.balance_multiplier,1.)

if __name__=='__main__':unittest.main()
