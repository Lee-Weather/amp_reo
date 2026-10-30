"""Temporal credit isolation, stationary counterexamples and explicit curriculum."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/amp_tools'))
from humanoid.amp.gait_guidance import GaitWindow, span_improvement
from preflight_gait_curriculum import configuration115, audit
from test_amp import library
from test_exp3 import definitions


class Exp115Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_cycle_beats_static_jitter_sliding_and_in_place_counterexamples(self):
        ns = configuration115()
        cfg = ns['X1AMPExp115Cfg']()
        lib = library()
        clip = lib.names.index(cfg.amp.standing_clip)
        standing = lib.state(torch.tensor([clip]), torch.tensor([cfg.amp.standing_frame/float(lib.fps[clip])]))[0, 13:42]
        self.assertTrue(audit(lib, cfg, standing)['verified'])

    def test_command_changes_and_episode_resets_require_a_new_full_window(self):
        window = GaitWindow(2, 3, 'cpu')
        q, feet, xy, heading = torch.zeros(2, 6), torch.zeros(2, 2), torch.zeros(2, 2), torch.tensor([[1., 0.]]*2)
        command = torch.full((2,), .3)
        for step in range(3):
            q[:] = step; xy[:, 0] = step*.006
            _, d, ready = window.update(q, feet, xy, heading, command)
        self.assertTrue(ready.all()); torch.testing.assert_close(d, torch.full((2,), .012))
        command[0] = 0.
        _, _, ready = window.update(q, feet, xy, heading, command)
        self.assertEqual(ready.tolist(), [False, True])
        window.reset(torch.tensor([1])); _, _, ready = window.update(q, feet, xy, heading, command)
        self.assertFalse(ready.any())

    def test_span_score_rejects_rest_and_excessive_motion(self):
        target = torch.ones(2, 8)
        torch.testing.assert_close(span_improvement(torch.zeros_like(target), target), torch.zeros(2))
        torch.testing.assert_close(span_improvement(target, target), torch.ones(2))
        self.assertTrue((span_improvement(3*target, target) < 0).all())

    def test_amp_ramp_and_both_branches_share_the_objective(self):
        ns = configuration115()
        a, b = ns['X1AMPExp115Cfg'](), ns['X1AMPExp115tCfg']()
        for key in ('env', 'commands', 'rewards', 'amp', 'control', 'sim', 'noise', 'domain_rand'):
            self.assertIs(getattr(type(a), key), getattr(type(b), key))
        namespace = {'torch': torch, 'X1AMPExp114Env': object}
        definitions('humanoid/envs/x1/x1_amp_exp115_env.py', namespace)
        env = namespace['X1AMPExp115Env'](); env.cfg = a
        for iteration, expected in ((0, 0.), (499, 0.), (500, 0.), (1000, .5), (1500, 1.), (2000, 1.)):
            env._style_iteration = iteration
            self.assertEqual(env._style_weight(), expected)
        ppo = ns['X1AMPExp115CfgPPO']()
        self.assertEqual((ppo.runner.num_steps_per_env, ppo.algorithm.num_learning_epochs, ppo.algorithm.num_mini_batches), (24, 2, 4))
        self.assertEqual((a.env.num_observations, a.env.num_actions, ppo.runner.max_iterations), (6468, 29, 1000))


if __name__ == '__main__': unittest.main()
