"""Production nominal reset/command/noise paths and real full-state continuation."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import torch
from humanoid.amp.exp212_transfer import SOURCE_SHA,expected_metadata,validate_configuration,restore_nominal_continuation
from humanoid.amp.nominal_startup import cohort_mask,training_metadata
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.amp.exp20_commands import SmoothCommands
from test_exp23 import fixture23,definitions,dictionary,configuration21
from test_exp210 import SOURCE


def configuration():
    ns=configuration21()
    for suffix in ('22','23','24','25','26','27','28','29','211','212'):
        definitions('humanoid/envs/x1/x1_amp_exp'+suffix+'_config.py',ns)
    cfg={**dictionary(ns['X1AMPExp212CfgPPO']()),**dictionary(ns['X1AMPExp212Cfg']())}
    cfg['seed']=5;cfg['init_member_classes']={}
    import math
    dt=cfg['sim']['dt']*cfg['control']['decimation']
    for k in ('push_interval','ext_force_interval'):cfg['domain_rand'][k]=math.ceil(cfg['domain_rand'][k+'_s']/dt)
    return cfg


class NominalStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def fresh(self):
        env,_=fixture23(self.source);env.gait_curriculum=Exp29Curriculum()
        env.checkpoint_training_metadata=lambda:expected_metadata(self.source['metadata'])['training']
        return Exp25AMPOnPolicyRunner(env,configuration())

    def test_config_preserves_rewards_controller_and_rejects_extra_changes(self):
        cfg=configuration();validate_configuration(self.source['config'],cfg)
        for key in ('rewards','control','policy','algorithm','noise','commands','env'):
            self.assertEqual(cfg[key],self.source['config'][key])
        for key in ('control','sim','rewards'):
            bad=copy.deepcopy(cfg);bad[key]['unexpected_change']=1
            with self.assertRaises(ValueError):validate_configuration(self.source['config'],bad)
        self.assertEqual(int(cohort_mask(torch.arange(4096),4096).sum()),1024)

    def test_real_production_reset_noisy_history_and_resample_paths(self):
        class Base:
            def _resample_commands(self,ids):
                self.standing_commands.requested[ids]=self._reset_commands if self._reset_commands is not None else 0.
            def compute_observations(self):
                self.obs_history.append(self._policy_frame()+.5)
                self.obs_buf=torch.cat(self.obs_history,-1)
        class Legged:
            @staticmethod
            def reset_idx(env,ids):
                env.captured_state=env._reset_reference.clone()
                env._resample_commands(ids)
        ns=dict(torch=torch,json=__import__('json'),X1AMPExp20Env=Base,LeggedRobot=Legged)
        definitions('humanoid/envs/x1/x1_amp_exp21_env.py',ns)
        ns.update(X1AMPExp211Env=ns['X1AMPExp21Env'],cohort_mask=cohort_mask,training_metadata=training_metadata)
        definitions('humanoid/envs/x1/x1_amp_exp212_env.py',ns)
        env=ns['X1AMPExp212Env'].__new__(ns['X1AMPExp212Env'])
        env.num_envs=8;env.device='cpu';env.dt=.02
        env.cfg=NS(amp=NS(nominal_startup_fraction=.25,standing_joint_noise=.03,ground_clearance=.002),commands=NS(native_speed=.5953696966))
        env.gait_curriculum=NS(weights={'rsi_probability':.5});env.evaluation_reset_mode=None
        env.standing_state=torch.zeros(71);env.standing_state[2]=.62;env.standing_state[6]=1
        env.hard_dof_limits=torch.tensor([[-2.,2.]]*29)
        env.reference_guide=NS(state=lambda phase:env.standing_state.expand(len(phase),-1).clone()+.01,
            sole_height=lambda state:torch.full((len(state),2),.002),nearest_phase=lambda q,dq:torch.full((len(q),),.2),
            history=lambda phase:torch.ones(len(phase),3,4))
        env.motion_library=NS(features=lambda state:torch.zeros(len(state),4))
        env._draw_commands=lambda n:torch.zeros(n,3)
        env._reset_commands=None;env.commands=torch.zeros(8,3);env.reset_standing=torch.zeros(8,dtype=torch.bool)
        env.standing_commands=SmoothCommands(8,'cpu',75,.02,2.,.5953696966)
        env.obs_history=[torch.ones(8,98)]*65
        env.amp_history=torch.zeros(8,3,4);env.amp_valid_history=torch.zeros(8,3,dtype=torch.bool)
        for name in ('zero_steps','stop_hold_steps','previous_moving','stop_pending','stand_gate','gait_phase','completed_cycles'):
            setattr(env,name,torch.zeros(8))
        torch.manual_seed(5)
        with contextlib.redirect_stdout(io.StringIO()):env.reset_idx(torch.arange(8))
        self.assertTrue(env.reset_standing[:2].all())
        torch.testing.assert_close(env.captured_state[:2],env.standing_state.expand(2,-1),rtol=0,atol=0)
        self.assertTrue((env.standing_commands.requested[:2,0]>.595).all())
        self.assertTrue((env.standing_commands.value[:2]==0).all())
        self.assertTrue((env.standing_commands.remaining[:2]==75).all())
        self.assertTrue((env.captured_state[2:,13:42]!=0).any())
        env._resample_commands(torch.arange(8))
        self.assertTrue((env.standing_commands.requested[:2,0]>.595).all())
        self.assertTrue((env.standing_commands.requested[2:]==0).all())
        env._policy_frame=lambda:torch.full((8,98),.125)
        with contextlib.redirect_stdout(io.StringIO()):env.compute_observations()
        self.assertEqual(env.obs_buf.shape,(8,6468))
        self.assertTrue((env.obs_buf[:2,-98:]==.125).all())
        self.assertTrue((env.obs_buf[2:,-98:]==.625).all())
        self.assertTrue((env.obs_buf[:2,:-98]==0).all())

    def test_actual_full_restore_update_and_save_reload(self):
        r=self.fresh()
        with contextlib.redirect_stdout(io.StringIO()):receipt=restore_nominal_continuation(r,SOURCE,SOURCE_SHA,13500,14500,1000)
        self.assertEqual(receipt['refinement'],'exp212_nominal_startup_v1')
        self.assertEqual(r.training_origin,self.source['training_origin'])
        for field,module in [('model_state_dict',r.actor_critic),('discriminator_state_dict',r.discriminator)]:
            for k,v in self.source[field].items():torch.testing.assert_close(module.state_dict()[k],v,rtol=0,atol=0)
        for field,opt in [('optimizer_state_dict',r.optimizer),('disc_optimizer_state_dict',r.disc_optimizer)]:
            for k,entry in self.source[field]['state'].items():
                for name,v in entry.items():torch.testing.assert_close(opt.state_dict()['state'][k][name],v,rtol=0,atol=0)
        with contextlib.redirect_stdout(io.StringIO()):r.learn(1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,13501);self.assertEqual(r.env.gait_curriculum.d_updates,13301)
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in self.source['model_state_dict'].items() if k.startswith('actor.')))
        for k,v in self.source['model_state_dict'].items():
            if k.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        for k,v in self.source['infos']['exp21']['anchor_state'].items():torch.testing.assert_close(r.policy_anchor.state_dict()[k],v,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_13501.pt'
            with contextlib.redirect_stdout(io.StringIO()):r.save(str(path));loaded=self.fresh();loaded.load(str(path))
            self.assertEqual(loaded.current_learning_iteration,13501)
            self.assertEqual(loaded._metadata(),r._metadata())


if __name__=='__main__':unittest.main()
