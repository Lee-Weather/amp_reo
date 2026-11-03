"""Actual8500 checkpoint restoration, update and save/load for the cloud entry."""
import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch

from humanoid.amp.exp25_runner import Exp25AMPOnPolicyRunner
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.scripts.resume_amp import restore_continuation
from test_exp23 import fixture23

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('EXP29_LONG_SOURCE',str(ROOT/'czy/data/exp2.9/model_8500.pt')))


class ExactResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(5)
        cls.source=torch.load(SOURCE,map_location='cpu',weights_only=False)
        cls.digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    def fresh(self):
        env,_=fixture23(self.source)
        env.gait_curriculum=Exp29Curriculum()
        env.checkpoint_training_metadata=lambda:copy.deepcopy(self.source['metadata']['training'])
        cfg=copy.deepcopy(self.source['config']);cfg['runner']['max_iterations']=5000
        return Exp25AMPOnPolicyRunner(env,cfg)

    def test_production_resume_preserves_every_state_then_updates_and_reloads(self):
        r=self.fresh()
        with contextlib.redirect_stdout(io.StringIO()):
            receipt=restore_continuation(r,SOURCE,self.digest,8500,13500,5000)
        self.assertEqual(receipt['source_iteration'],8500)
        self.assertEqual(r.training_origin,self.source['training_origin'])
        self.assertEqual(r.env.gait_curriculum.state_dict(),self.source['infos']['exp21']['learning_state']['curriculum'])
        for field,module in [('model_state_dict',r.actor_critic),('discriminator_state_dict',r.discriminator)]:
            for name,value in self.source[field].items():torch.testing.assert_close(module.state_dict()[name],value,rtol=0,atol=0)
        for field,opt in [('optimizer_state_dict',r.optimizer),('disc_optimizer_state_dict',r.disc_optimizer)]:
            for key,entry in self.source[field]['state'].items():
                for name,value in entry.items():torch.testing.assert_close(opt.state_dict()['state'][key][name],value,rtol=0,atol=0)
        for name,value in self.source['infos']['exp21']['anchor_state'].items():torch.testing.assert_close(r.policy_anchor.state_dict()[name],value,rtol=0,atol=0)
        with contextlib.redirect_stdout(io.StringIO()):r.learn(1,init_at_random_ep_len=False)
        self.assertEqual(r.current_learning_iteration,8501)
        self.assertEqual(r.env.gait_curriculum.d_updates,8301)
        self.assertTrue(any(not torch.equal(v,r.actor_critic.state_dict()[k]) for k,v in self.source['model_state_dict'].items() if k.startswith('actor.')))
        for name,value in self.source['model_state_dict'].items():
            if name.startswith(('long_history.','state_estimator.','log_std')):torch.testing.assert_close(r.actor_critic.state_dict()[name],value,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_8501.pt'
            with contextlib.redirect_stdout(io.StringIO()):
                r.save(str(path));loaded=self.fresh();loaded.load(str(path))
            self.assertEqual(loaded.current_learning_iteration,8501)
            self.assertEqual(loaded.env.gait_curriculum.state_dict(),r.env.gait_curriculum.state_dict())

    def test_bad_hash_rejected_before_runner_mutation(self):
        r=self.fresh()
        with self.assertRaisesRegex(ValueError,'SHA256'):
            restore_continuation(r,SOURCE,'0'*64,8500,13500,5000)
        self.assertEqual(r.current_learning_iteration,0)
        self.assertFalse(r.optimizer.state)


if __name__=='__main__':unittest.main()
