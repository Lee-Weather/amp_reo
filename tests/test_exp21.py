"""Actual freeze/update/save paths and disjoint, resumable performance gates."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch

from humanoid.amp.gait_curriculum import GaitCurriculum
from humanoid.amp.conservative_runner import ConservativeAMPOnPolicyRunner
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.exp20_commands import SmoothCommands
from test_exp20 import configuration20
from test_exp117 import definitions,grounded_library
from preflight_reference_transfer import dictionary
from preflight_phase_refinement import environment

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP21_SOURCE_MODEL',str(ROOT/'czy/data/exp1.17/model_7000.pt')))


def configuration21():
    ns=configuration20()
    definitions('humanoid/envs/x1/x1_amp_exp21_config.py',ns)
    return ns


def fixture21(source):
    settings=configuration21()['X1AMPExp21Cfg']()
    cfg=dictionary(configuration21()['X1AMPExp21CfgPPO']());cfg.update(dictionary(settings))
    lib=MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized',
        ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf',
        source['metadata']['dof_names'],settings.amp.key_body_names,dt=.02,history_steps=3,
        weights=settings.amp.motion_weights,ground_clearance=.002)
    env=environment(lib,settings,source['metadata'])
    env.checkpoint_training_metadata=lambda:dict(curriculum='performance_gated_native_v1',
        reference_source_sha256=source['metadata']['training']['reference_source_sha256'])
    env.gait_curriculum=GaitCurriculum()
    env.checkpoint_learning_state=lambda:dict(curriculum=env.gait_curriculum.state_dict())
    env.restore_learning_state=lambda value:env.gait_curriculum.load_state_dict(value['curriculum'])
    env.discriminator_training_enabled=lambda:env.gait_curriculum.weights['amp']>0
    env.complete_training_rollout=lambda iteration:None
    env.sample_amp_expert_by_category=lambda labels:lib.sample(len(labels))
    step=env.step
    def wrapped(actions):
        obs,critic,reward,done,info=step(actions)
        info.update(amp_valid=torch.tensor([True,False,True,True]),amp_moving=torch.tensor([True,False,False,True]),
                    amp_style_multiplier=torch.zeros(4))
        return obs,critic,reward,done,info
    env.step=wrapped
    return env,cfg


class CurriculumTests(unittest.TestCase):
    def test_disjoint_gates_do_not_promote_on_mixed_average(self):
        c=GaitCurriculum()
        for step in (50,100,150):
            c.add([[200,200*2,200,200],[800,800*20,0,0]],10000,8000)
            r=c.finish(step)
            self.assertFalse(r['qualified']);self.assertEqual(r['next_stage'],0)
        self.assertEqual(c.last_window['standing']['mean_seconds'],2)
        self.assertEqual(c.last_window['rsi']['mean_seconds'],20)
        self.assertEqual(c.pending,[[0.,0.,0.,0.],[0.,0.,0.,0.]])

    def test_resume_keeps_pending_window_and_changes_one_dimension(self):
        c=GaitCurriculum()
        c.add([[100,1600,10,0],[200,3400,10,0]],10000,6000)
        d=GaitCurriculum();d.load_state_dict(c.state_dict())
        self.assertEqual(c.state_dict(),d.state_dict())
        for i in (50,100,150):
            if i>50: d.add([[100,1600,10,0],[200,3400,10,0]],10000,6000)
            report=d.finish(i)
        self.assertEqual(report['next_stage'],1)
        before=d.weights;d.commit(1);after=d.weights
        self.assertEqual(sum(before[k]!=after[k] for k in before),1)
        for i in (200,250):
            d.add([[100,400,100,100],[200,800,200,200]],10000,6000)
            report=d.finish(i)
        self.assertEqual(report['next_stage'],0)


class ConservativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def runner(self):
        env,cfg=fixture21(self.source)
        r=ConservativeAMPOnPolicyRunner(env,copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):r.warm_start(str(SOURCE))
        return r

    def test_critic_preheat_changes_only_critic_and_pauses_D(self):
        r=self.runner();before=copy.deepcopy(r.actor_critic.state_dict())
        disc=copy.deepcopy(r.discriminator.state_dict())
        batch=r.collect_rollout();metrics=r.update_policy(batch)
        r.update_discriminator(batch['amp'],batch['amp_valid'],batch['amp_moving'])
        after=r.actor_critic.state_dict()
        self.assertTrue(any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('critic.')))
        for k,v in before.items():
            if not k.startswith('critic.'):self.assertTrue(torch.equal(v,after[k]),k)
        for k,v in disc.items():self.assertTrue(torch.equal(v,r.discriminator.state_dict()[k]),k)
        self.assertFalse(r.disc_optimizer.state)
        self.assertGreater(r.replay.size,0)
        self.assertEqual(metrics['policy_updates'],0)
        self.assertEqual(metrics['source_action_rms_delta'],0)

    def test_policy_unfreezes_but_std_stays_fixed_and_resume_keeps_anchor(self):
        r=self.runner();r.current_learning_iteration=50
        before=copy.deepcopy(r.actor_critic.state_dict())
        batch=r.collect_rollout();r.update_policy(batch)
        after=r.actor_critic.state_dict()
        self.assertTrue(any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('actor.')))
        self.assertTrue(torch.equal(before['log_std'],after['log_std']))
        r.env.gait_curriculum.add([[10,170,0,0],[20,350,0,0]],2000,1000)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model_50.pt'
            with contextlib.redirect_stdout(io.StringIO()):r.save(str(path))
            loaded=self.runner();loaded.load(str(path))
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())
            for k,v in r.policy_anchor.state_dict().items():self.assertTrue(torch.equal(v,loaded.policy_anchor.state_dict()[k]))
            self.assertEqual(loaded.current_learning_iteration,50)

    def test_native_rsi_reset_preserves_full_state_phase_and_history(self):
        env,cfg=fixture21(self.source)
        lib=env.motion_library
        correction=__import__('json').loads((ROOT/'humanoid/amp/reference_data/exp20_native_correction.json').read_text())
        reference=NativeCycle(lib,contact_correction=correction)
        class ResetBase:
            @staticmethod
            def reset_idx(target,ids):
                target.captured=target._reset_reference.clone()
                target.standing_commands.requested[ids]=target._reset_commands
        ns=dict(torch=torch,X1AMPExp20Env=object,LeggedRobot=ResetBase,GaitCurriculum=GaitCurriculum)
        definitions('humanoid/envs/x1/x1_amp_exp21_env.py',ns)
        target=object.__new__(ns['X1AMPExp21Env'])
        count=32
        target.cfg=configuration21()['X1AMPExp21Cfg']()
        target.gait_curriculum=GaitCurriculum()
        target.evaluation_reset_mode='rsi'
        target.num_envs,target.device=count,'cpu'
        target.reference_guide,target.motion_library=reference,lib
        target.standing_state=torch.tensor(self.source['metadata']['control']['standing_state'])
        target.hard_dof_limits=torch.stack((lib.fk.lower,lib.fk.upper),-1)
        target._draw_commands=lambda n:torch.zeros(n,3)
        target.standing_commands=SmoothCommands(count,'cpu',75,.02)
        target.reset_standing=torch.ones(count,dtype=torch.bool)
        target.commands=torch.zeros(count,3)
        target.obs_history=[torch.ones(count,98) for _ in range(66)]
        target.amp_history=torch.zeros(count,3,86)
        target.amp_valid_history=torch.zeros(count,3,dtype=torch.bool)
        for name in ('zero_steps','stop_hold_steps','previous_moving','stop_pending','stand_gate','gait_phase','completed_cycles'):
            setattr(target,name,torch.zeros(count))
        ids=torch.arange(count);target.reset_idx(ids)
        expected=reference.state(target.gait_phase)
        torch.testing.assert_close(target.captured[:,3:],expected[:,3:])
        torch.testing.assert_close(target.amp_history,reference.history(target.gait_phase))
        self.assertFalse(target.standing_commands.remaining.any())
        self.assertFalse(target.reset_standing.any())
        torch.testing.assert_close(target.commands[:,0],torch.full((count,),target.cfg.commands.native_speed))
        self.assertTrue(all(not frame.any() for frame in target.obs_history))


if __name__=='__main__':unittest.main()
