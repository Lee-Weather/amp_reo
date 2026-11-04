"""Contact populations, production shape hook and real full-state continuation."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

from humanoid.amp.contact_domain import sample_offsets,set_contact_offsets
from humanoid.amp.exp211_transfer import SOURCE_SHA,expected_metadata,validate_configuration,restore_contact_continuation
from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from test_exp23 import fixture23,definitions,dictionary,configuration21
from test_exp210 import SOURCE


def configuration():
    ns=configuration21()
    for suffix in ('22','23','24','25','26','27','28','29','211'):
        definitions('humanoid/envs/x1/x1_amp_exp'+suffix+'_config.py',ns)
    cfg={**dictionary(ns['X1AMPExp211CfgPPO']()),**dictionary(ns['X1AMPExp211Cfg']())}
    cfg['seed']=5;cfg['init_member_classes']={}
    dt=cfg['sim']['dt']*cfg['control']['decimation']
    import math
    for k in ('push_interval','ext_force_interval'):cfg['domain_rand'][k]=math.ceil(cfg['domain_rand'][k+'_s']/dt)
    return cfg


class ContactContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def fresh(self):
        env,_=fixture23(self.source);env.gait_curriculum=Exp29Curriculum()
        env.checkpoint_training_metadata=lambda:expected_metadata(self.source['metadata'])['training']
        return Exp25AMPOnPolicyRunner(env,configuration())

    def test_only_contact_domain_changes_and_bad_reward_rejected(self):
        cfg=configuration();validate_configuration(self.source['config'],cfg)
        self.assertEqual(cfg['rewards'],self.source['config']['rewards'])
        self.assertEqual(cfg['domain_rand']['friction_range'],[.65,.95])
        wrong=copy.deepcopy(cfg);wrong['rewards']['scales']['corridor_yaw']=-.30
        with self.assertRaisesRegex(ValueError,'Unexpected'):validate_configuration(self.source['config'],wrong)

    def test_production_shape_hook_preserves_friction_and_population(self):
        class Base:
            def _process_rigid_shape_props(self,props,env_id):return props
        ns=dict(torch=torch,sample_offsets=sample_offsets,set_contact_offsets=set_contact_offsets,X1AMPExp29Env=Base)
        definitions('humanoid/envs/x1/x1_amp_exp211_env.py',ns)
        env=ns['X1AMPExp211Env'].__new__(ns['X1AMPExp211Env']);env.num_envs=4096
        env.cfg=SimpleNamespace(domain_rand=SimpleNamespace(randomize_contact_offset=True,contact_offset_range=[.001,.020]))
        for index,expected in ((0,.001),(2048,.01),(3072,None)):
            props=[SimpleNamespace(friction=.8,restitution=0.,contact_offset=.01,rest_offset=0.) for _ in range(2)]
            out=env._process_rigid_shape_props(props,index)
            self.assertTrue(all(p.friction==.8 and p.restitution==0. and p.rest_offset==0. for p in out))
            self.assertTrue(all(.001-1e-8<=p.contact_offset<=.02+1e-8 for p in out))
            if expected is not None:self.assertAlmostEqual(out[0].contact_offset,expected,places=7)
        self.assertEqual(int((env.contact_offset_values<=.00100001).sum()),2048)
        self.assertEqual(int((env.contact_offset_values==.01).sum()),1024)
        with self.assertRaises(ValueError):set_contact_offsets(props,0.)

    def test_full13500_state_then_update_and_save_reload(self):
        r=self.fresh()
        with contextlib.redirect_stdout(io.StringIO()):receipt=restore_contact_continuation(r,SOURCE,SOURCE_SHA,13500,14500,1000)
        self.assertEqual(receipt['refinement'],'exp211_contact_only_v1')
        self.assertEqual(r._metadata(),expected_metadata(self.source['metadata']))
        self.assertEqual(r.training_origin,self.source['training_origin'])
        self.assertEqual(r.env.gait_curriculum.state_dict(),self.source['infos']['exp21']['learning_state']['curriculum'])
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
