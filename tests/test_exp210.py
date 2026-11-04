"""Real13500 yaw-only full-state restoration, update and save/reload contracts."""
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
import torch

from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.amp.exp210_transfer import SOURCE_SHA, expected_metadata, validate_configuration, restore_yaw_continuation
from test_exp23 import fixture23, definitions, dictionary, configuration21

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP210_SOURCE_MODEL',str(ROOT/'czy/data/exp2.9/model_13500.pt')))


def configuration():
    ns=configuration21()
    for suffix in ('22','23','24','25','26','27','28','29','210'):
        definitions('humanoid/envs/x1/x1_amp_exp'+suffix+'_config.py',ns)
    cfg={**dictionary(ns['X1AMPExp210CfgPPO']()),**dictionary(ns['X1AMPExp210Cfg']())}
    cfg['seed']=5
    dt=cfg['sim']['dt']*cfg['control']['decimation']
    # Derived by LeggedRobot._parse_cfg after real environment construction.
    cfg['domain_rand']['push_interval']=math.ceil(cfg['domain_rand']['push_interval_s']/dt)
    cfg['domain_rand']['ext_force_interval']=math.ceil(cfg['domain_rand']['ext_force_interval_s']/dt)
    cfg['init_member_classes']={}
    return cfg


class YawContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)

    def fresh(self):
        env,_=fixture23(self.source)
        env.gait_curriculum=Exp29Curriculum()
        env.checkpoint_training_metadata=lambda:expected_metadata(self.source['metadata'])['training']
        return Exp25AMPOnPolicyRunner(env,configuration())

    def test_production_configuration_changes_only_yaw_and_run_identity(self):
        cfg=configuration()
        validate_configuration(self.source['config'],cfg)
        self.assertEqual(cfg['rewards']['scales']['corridor_yaw'],-.30)
        for field in ('control','sim','amp','policy','algorithm','discriminator','domain_rand'):
            self.assertEqual(cfg[field],self.source['config'][field])
        wrong=copy.deepcopy(cfg);wrong['control']['action_scale']=.3
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            validate_configuration(self.source['config'],wrong)

    def test_restore_every_tensor_and_adam_then_learn_save_reload(self):
        runner=self.fresh()
        with contextlib.redirect_stdout(io.StringIO()):
            receipt=restore_yaw_continuation(runner,SOURCE,SOURCE_SHA,13500,14500,1000)
        self.assertEqual(receipt['refinement'],'exp210_yaw_only_v1')
        self.assertEqual(runner.current_learning_iteration,13500)
        self.assertEqual(runner.session_start_iteration,13500)
        self.assertEqual(runner.training_origin,self.source['training_origin'])
        self.assertEqual(runner._metadata(),expected_metadata(self.source['metadata']))
        self.assertEqual(runner.env.gait_curriculum.state_dict(),self.source['infos']['exp21']['learning_state']['curriculum'])
        for field,module in [('model_state_dict',runner.actor_critic),('discriminator_state_dict',runner.discriminator)]:
            for k,v in self.source[field].items():torch.testing.assert_close(module.state_dict()[k],v,rtol=0,atol=0)
        for field,opt in [('optimizer_state_dict',runner.optimizer),('disc_optimizer_state_dict',runner.disc_optimizer)]:
            for k,entry in self.source[field]['state'].items():
                for name,value in entry.items():torch.testing.assert_close(opt.state_dict()['state'][k][name],value,rtol=0,atol=0)
        for k,v in self.source['infos']['exp21']['anchor_state'].items():
            torch.testing.assert_close(runner.policy_anchor.state_dict()[k],v,rtol=0,atol=0)
        with contextlib.redirect_stdout(io.StringIO()):runner.learn(1,init_at_random_ep_len=False)
        self.assertEqual(runner.current_learning_iteration,13501)
        self.assertEqual(runner.env.gait_curriculum.d_updates,13301)
        for k,v in self.source['model_state_dict'].items():
            if k.startswith(('long_history.','state_estimator.','log_std')):
                torch.testing.assert_close(runner.actor_critic.state_dict()[k],v,rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,runner.actor_critic.state_dict()[k]) for k,v in self.source['model_state_dict'].items() if k.startswith('actor.')))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_13501.pt'
            with contextlib.redirect_stdout(io.StringIO()):
                runner.save(str(path));loaded=self.fresh();loaded.load(str(path))
            self.assertEqual(loaded.current_learning_iteration,13501)
            self.assertEqual(loaded._metadata(),runner._metadata())
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),runner.env.gait_curriculum.state_dict())

    def test_invalid_hash_rejected_before_state_changes(self):
        runner=self.fresh()
        with self.assertRaisesRegex(ValueError,'SHA256'):
            restore_yaw_continuation(runner,SOURCE,'0'*64,13500,14500,1000)
        self.assertEqual(runner.current_learning_iteration,0)
        self.assertFalse(runner.optimizer.state)


if __name__=='__main__':unittest.main()
