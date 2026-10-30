"""Phase reward direction, reset isolation and AMP-preserving reward migration."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/amp_tools'))
from preflight_phase_refinement import configuration116, environment, reward_audit
from preflight_reference_transfer import dictionary
from humanoid.amp.phase_rewards import phase_progress, phase_velocity_credit, phase_quality, phase_support_credit
from humanoid.amp.runner import AMPOnPolicyRunner
from test_amp import library
from test_exp3 import definitions


class Exp116Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = configuration116(); cls.lib = library()

    def test_large_error_still_has_directional_feedback_and_static_zero(self):
        scale = torch.ones(2, 6)
        target = torch.ones(2, 6)*10
        previous = torch.zeros_like(target)
        actual = previous.clone(); actual[0] += .01; actual[1] -= .01
        progress = phase_progress(previous, actual, target, scale, .02)
        self.assertGreater(float(progress[0]), 0.)
        self.assertLess(float(progress[1]), 0.)
        torch.testing.assert_close(phase_progress(actual, actual, target, scale, .02), torch.zeros(2))
        self.assertTrue((phase_quality(actual, target, scale) > .05).all())

    def test_velocity_and_support_reject_static_wrong_direction_and_excess_speed(self):
        target = torch.ones(3, 6); scale = torch.ones_like(target)
        actual = torch.stack((torch.zeros(6), torch.ones(6), -torch.ones(6)))
        credit = phase_velocity_credit(actual, target, scale)
        self.assertEqual(float(credit[0]), 0.)
        self.assertGreater(float(credit[1]), 0.)
        self.assertLess(float(credit[2]), 0.)
        self.assertLess(float(phase_velocity_credit(target*10, target, scale).mean()), 0.)
        height = torch.tensor([[.03, 0.]]*3); lift = height.clone()
        force = torch.tensor([[0., 100.], [100., 0.], [100., 100.]])
        score = phase_support_credit(height, force, lift, torch.ones(3))
        self.assertGreater(float(score[0]), float(score[1]))
        self.assertGreater(float(score[0]), float(score[2]))
        torch.testing.assert_close(phase_support_credit(height, force, lift, torch.zeros(3)), torch.zeros(3))

    def test_transition_gate_skips_reset_and_command_change_but_not_two_second_history(self):
        class Parent:
            def reset_idx(self, ids): pass
            def step(self, actions): return actions
        ns = dict(torch=torch, X1AMPExp115Env=Parent)
        definitions('humanoid/envs/x1/x1_amp_exp116_env.py', ns)
        env = ns['X1AMPExp116Env']()
        env.previous_guidance_command = torch.tensor([.3, .3, 0.])
        env.commands = torch.tensor([[.3, 0., 0.], [.3, 0., 0.], [.3, 0., 0.]])
        env.guidance_transition_valid = torch.ones(3, dtype=torch.bool)
        self.assertEqual(env._transition_valid().tolist(), [True, True, False])
        env.reset_idx(torch.tensor([0]))
        self.assertEqual(env._transition_valid().tolist(), [False, True, False])

    def test_real_reference_beats_static_wrong_phase_and_fast_counterexamples(self):
        cfg = self.ns['X1AMPExp116Cfg']()
        i = self.lib.names.index(cfg.amp.standing_clip)
        standing = self.lib.state(torch.tensor([i]), torch.tensor([cfg.amp.standing_frame/float(self.lib.fps[i])]))[0, 13:42]
        self.assertTrue(reward_audit(self.lib, cfg, standing)['verified'])

    def settings(self, source=False):
        cfg = dictionary(self.ns['X1AMPExp116CfgPPO']()); settings = self.ns['X1AMPExp116Cfg']()
        cfg.update(dictionary(settings))
        # Small network for migration contract tests, with the real98-D DH layout.
        cfg['policy'].update(actor_hidden_dims=[32], critic_hidden_dims=[32], state_estimator_hidden_dims=[32])
        cfg['discriminator'].update(hidden_dims=[32], replay_size=64, batch_size=8, num_updates=2)
        cfg['runner']['num_steps_per_env'] = 4
        if source:
            cfg['runner'].update(experiment_name='x1_amp_exp115t', require_reward_transfer=False)
            cfg['amp'].pop('fixed_style_weight')
        return cfg, settings

    def test_original_control_budget_and_commands_remain_exact(self):
        source, target = self.ns['X1AMPExp115Cfg'], self.ns['X1AMPExp116Cfg']
        for section in ('env', 'control', 'commands', 'asset', 'sim', 'noise', 'domain_rand'):
            self.assertIs(getattr(source, section), getattr(target, section))
        cfg, settings = self.settings()
        ns = dict(torch=torch, X1AMPExp115Env=object)
        definitions('humanoid/envs/x1/x1_amp_exp116_env.py', ns)
        env = ns['X1AMPExp116Env'](); env.cfg = settings
        for iteration in (0, 999, 1000, 1500, 2000):
            env._style_iteration = iteration
            self.assertEqual(env._style_weight(), .5)
        self.assertEqual(settings.rewards.scales.reference_amplitude, 0.)
        self.assertEqual(settings.rewards.scales.net_displacement, 0.)
        self.assertEqual((settings.env.num_observations, settings.env.num_actions), (6468, 29))

    def source(self, folder):
        cfg, settings = self.settings(True)
        env = environment(self.lib, settings)
        env.checkpoint_training_metadata = lambda: {'curriculum': 'normalized_forward_guidance_v1'}
        runner = AMPOnPolicyRunner(env, cfg)
        with contextlib.redirect_stdout(io.StringIO()): runner.learn(1)
        runner.current_learning_iteration = 1000
        path = Path(folder)/'source.pt'
        with contextlib.redirect_stdout(io.StringIO()): runner.save(str(path))
        return runner, path

    def target(self, path):
        cfg, settings = self.settings()
        cfg['runner']['policy_transfer_source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return AMPOnPolicyRunner(environment(self.lib, settings), cfg)

    def test_preserves_amp_moments_rebuilds_critic_and_resumes_after_save(self):
        with tempfile.TemporaryDirectory() as folder:
            source, path = self.source(folder); target = self.target(path)
            fresh = copy.deepcopy(target.actor_critic.critic.state_dict())
            with contextlib.redirect_stdout(io.StringIO()): target.warm_start(str(path))
            self.assertFalse(target.optimizer.state); self.assertEqual(target.replay.size, 0)
            self.assertEqual(target.current_learning_iteration, 1000)
            for k, v in fresh.items(): self.assertTrue(torch.equal(v, target.actor_critic.critic.state_dict()[k]))
            for k, v in source.actor_critic.state_dict().items():
                if not k.startswith('critic.'): self.assertTrue(torch.equal(v, target.actor_critic.state_dict()[k]))
            for k, v in source.discriminator.state_dict().items(): self.assertTrue(torch.equal(v, target.discriminator.state_dict()[k]))
            for original, retained in zip(source.disc_optimizer.state.values(), target.disc_optimizer.state.values()):
                for k, v in original.items(): self.assertTrue(torch.equal(v, retained[k]))
            for iteration, cap in ((1000, 1e-5), (1099, 1e-5), (1100, 1e-4)):
                target.current_learning_iteration = iteration; self.assertEqual(target._policy_learning_rate_cap(), cap)
            target.current_learning_iteration = 1000
            with contextlib.redirect_stdout(io.StringIO()): target.learn(1)
            self.assertEqual(target.current_learning_iteration, 1001)
            self.assertTrue(target.optimizer.state)
            self.assertEqual(min(float(v['step']) for v in target.disc_optimizer.state.values()), 4.)
            saved = str(Path(folder)/'continued.pt')
            with contextlib.redirect_stdout(io.StringIO()): target.save(saved)
            resumed = self.target(path)
            with contextlib.redirect_stdout(io.StringIO()): resumed.load(saved)
            self.assertEqual(resumed.current_learning_iteration, 1001)
            self.assertEqual(resumed.training_origin, target.training_origin)
            self.assertEqual(resumed._policy_learning_rate_cap(), 1e-5)

    def test_invalid_source_rejected_before_any_weights_are_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            _, path = self.source(folder)
            state = torch.load(path, weights_only=False)
            for change in ('sha', 'normalization', 'disc_moments', 'curriculum', 'iteration'):
                with self.subTest(change=change):
                    altered = copy.deepcopy(state)
                    if change == 'normalization': altered['config']['normalization']['clip_observations'] += 1.
                    if change == 'disc_moments': next(iter(altered['disc_optimizer_state_dict']['state'].values()))['exp_avg'][0] = float('nan')
                    if change == 'curriculum': altered['metadata']['training']['curriculum'] = 'wrong'
                    if change == 'iteration': altered['iter'] = 999
                    bad = Path(folder)/'bad.pt'; torch.save(altered, bad)
                    target = self.target(bad)
                    before = copy.deepcopy(target.actor_critic.state_dict())
                    if change == 'sha': target.cfg['policy_transfer_source_sha256'] = '0'*64
                    with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()): target.warm_start(str(bad))
                    for k, v in before.items(): self.assertTrue(torch.equal(v, target.actor_critic.state_dict()[k]))
                    self.assertFalse(target.optimizer.state); self.assertFalse(target.disc_optimizer.state)


if __name__ == '__main__': unittest.main()
