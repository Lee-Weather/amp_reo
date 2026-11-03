"""Full overnight learning entry, preserved source and bounded stage progression."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from humanoid.amp.exp27_curriculum import Exp27Curriculum
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from test_exp23 import fixture23,definitions,dictionary,configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP27_SOURCE_MODEL',str(ROOT/'czy/data/exp2.6/model_5100.pt')))

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture23(self.source)
        ns=configuration21()
        for n in (22,23,24,25,26,27):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        cfg.update(dictionary(ns['X1AMPExp27CfgPPO']()))
        cfg['discriminator']['batch_size']=32
        env.gait_curriculum=Exp27Curriculum()
        env.checkpoint_training_metadata=lambda:dict(curriculum='contact_balance_low_amp_v1',
            reference_source_sha256=self.source['metadata']['training']['reference_source_sha256'])
        r=Exp25AMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_preserves_full_policy_adam_anchor_and_iteration(self):
        r=self.runner()
        self.assertEqual(r.current_learning_iteration,5100)
        self.assertEqual(r.training_origin['new_iteration'],5100)
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
        self.assertEqual(r.env.gait_curriculum.window_start,5100)
        self.assertEqual(r.env.gait_curriculum.baseline,self.source['infos']['exp21']['learning_state']['curriculum']['last_window']['pooled100'])

    def test_complete_learn_trains_d_with_zero_amp_reward_and_resumes(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict());d=copy.deepcopy(r.discriminator.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,5101)
        self.assertEqual(r.env.gait_curriculum.d_updates,4901)
        self.assertEqual(r.env.gait_curriculum.weights['amp'],.01)
        self.assertTrue(r.disc_optimizer.state)
        self.assertTrue(any(not torch.equal(v,r.discriminator.state_dict()[k]) for k,v in d.items()))
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in before.items() if k.startswith('actor.')))
        for k,v in before.items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model_5101.pt')
            with contextlib.redirect_stdout(io.StringIO()):r.save(path)
            loaded=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):loaded.load(path)
            self.assertEqual(loaded.current_learning_iteration,5101)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            self.assertEqual(loaded.best_record,r.best_record)
            self.assertTrue(loaded.disc_optimizer.state)

    def test_wall_limit_saves_before_exit_without_update(self):
        r=self.runner();r.wall_started_at-=20000
        with tempfile.TemporaryDirectory() as tmp:
            r.log_dir=tmp
            with contextlib.redirect_stdout(io.StringIO()):r.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            self.assertEqual(r.current_learning_iteration,5100)
            state=torch.load(Path(tmp)/'model_5100.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['infos']['exp24']['stop_reason'],'wall_time_limit_5_5_hours')

class BalanceTests(unittest.TestCase):
    def test_corrective_costs_grow_beyond_tracking_peak_and_are_symmetric(self):
        from humanoid.amp.exp27_balance import corridor_costs
        v=torch.tensor([0.,.15,.3,.6]);w=torch.tensor([0.,.5,1.,2.])
        lateral,yaw=corridor_costs(v,w,torch.zeros(4))
        self.assertEqual(float(lateral[0]),0);self.assertTrue(torch.all(lateral[1:]>lateral[:-1]))
        torch.testing.assert_close(lateral,yaw)
        negative=corridor_costs(-v,-w,torch.zeros(4))
        torch.testing.assert_close(lateral,negative[0]);torch.testing.assert_close(yaw,negative[1])
        self.assertEqual(float(corridor_costs(v,w,w)[1].sum()),0)

    def test_hold_swing_and_amp_and_ramp_only_balance(self):
        c=Exp27Curriculum();c.last_iteration=5100
        self.assertEqual(c.swing_multiplier,2);self.assertEqual(c.weights['amp'],.01)
        self.assertEqual(c.balance_multiplier,0)
        c.last_iteration=5200;self.assertEqual(c.balance_multiplier,.5)
        c.last_iteration=5300;self.assertEqual(c.balance_multiplier,1.)
        self.assertEqual(c.swing_multiplier,2.)

    def test_actual_rigid_shape_callback_randomizes_only_friction_not_bounce(self):
        import ast
        from types import SimpleNamespace
        ns=configuration21()
        for n in (22,23,24,25,26,27):definitions('humanoid/envs/x1/x1_amp_exp'+str(n)+'_config.py',ns)
        old=dictionary(ns['X1AMPExp26Cfg']())['domain_rand'];new=dictionary(ns['X1AMPExp27Cfg']())['domain_rand']
        self.assertEqual({k for k in old if old[k]!=new[k]}, {'randomize_friction','friction_range','restitution_range'})
        tree=ast.parse((ROOT/'humanoid/envs/base/legged_robot.py').read_text())
        node=next(x for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='LeggedRobot' for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_process_rigid_shape_props')
        module=ast.Module(body=[node],type_ignores=[])
        scope=dict(torch=torch,torch_rand_float=lambda low,high,shape,device:low+(high-low)*torch.rand(shape,device=device))
        exec(compile(module,'rigid-shape-callback','exec'),scope)
        env=SimpleNamespace(cfg=ns['X1AMPExp27Cfg'](),num_envs=4,env_frictions=torch.zeros(4,1))
        for i in range(4):
            props=[SimpleNamespace(friction=1.,restitution=.0) for _ in range(2)]
            scope['_process_rigid_shape_props'](env,props,i)
            self.assertTrue(.65<=float(props[0].friction)<=.95)
            self.assertEqual(float(props[0].restitution),0.)
        self.assertTrue(torch.all((env.env_frictions>=.65)&(env.env_frictions<=.95)))

    def test_environment_uses_valid_motion_gate_and_ramped_negative_reward(self):
        from humanoid.amp.exp27_balance import corridor_costs
        class Base:pass
        ns=dict(X1AMPExp26Env=Base,Exp27Curriculum=Exp27Curriculum,corridor_costs=corridor_costs)
        definitions('humanoid/envs/x1/x1_amp_exp27_env.py',ns)
        env=object.__new__(ns['X1AMPExp27Env'])
        env.base_lin_vel=torch.tensor([[0.,.3,0.],[0.,.3,0.]])
        env.base_ang_vel=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
        env.commands=torch.zeros(2,3);env._reference_valid=lambda:torch.tensor([True,False])
        env.gait_curriculum=Exp27Curriculum();env.gait_curriculum.last_iteration=5200
        self.assertGreater(float(env._reward_corridor_lateral()[0]),0)
        self.assertEqual(float(env._reward_corridor_lateral()[1]),0)
        self.assertEqual(float(env._reward_corridor_yaw()[1]),0)
        self.assertLess(float(env._reward_corridor_yaw()[0])*(-.10*.02),0)

if __name__=='__main__':unittest.main()
