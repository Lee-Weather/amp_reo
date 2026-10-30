"""Phase task integration, exact inference transfer and training-state isolation."""
import contextlib
import copy
import hashlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from humanoid.amp.motion import URDFKinematics
from humanoid.amp.phase_reference import PhaseReference
from reference_task_fixtures import configuration114, cpu_runner
from test_amp import library, URDF
from test_exp3 import definitions


class Exp114Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lib = library()

    def test_production_dimensions_and_common_constraints_are_explicit(self):
        ns = configuration114("114")
        cfg, ppo = ns["X1AMPExp114Cfg"], ns["X1AMPExp114CfgPPO"]
        old = ns["X1AMPExp113Cfg"]
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions), (6468, 101, 29))
        self.assertEqual((cfg.env.frame_stack, cfg.env.short_frame_stack, cfg.env.num_single_obs), (66, 5, 98))
        for key in ("control", "commands", "asset", "sim", "noise", "domain_rand"):
            self.assertIs(getattr(cfg, key), getattr(old, key))
        self.assertEqual((cfg.rewards.curriculum_initial_stand_weight, cfg.rewards.scales.stand_pose,
                          cfg.rewards.scales.raw_target_violation), (1., -1.2, -2.))
        self.assertEqual((old.rewards.curriculum_initial_stand_weight, old.rewards.scales.stand_pose), (.1, -.6))
        self.assertEqual((ppo.runner.num_steps_per_env, ppo.algorithm.num_learning_epochs,
                          ppo.algorithm.num_mini_batches, ppo.discriminator.num_updates), (24, 2, 4, 4))
        self.assertEqual(ppo.algorithm.state_estimator_obs_index, 98)
        self.assertEqual(ppo.runner.warm_start_mode, "policy_append_phase")

    def test_transfer_preserves_old_inference_for_arbitrary_phase_and_fresh_heads(self):
        source = cpu_runner(self.lib, source=True)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            source.learn(2); source.current_learning_iteration = 6000
            path = Path(td)/"source.pt"; source.save(path)
            target = cpu_runner(self.lib, phase=True)
            critic = copy.deepcopy(target.actor_critic.critic.state_dict())
            discriminator = copy.deepcopy(target.discriminator.state_dict())
            target.warm_start(path, hashlib.sha256(path.read_bytes()).hexdigest())
            old_obs = torch.randn(8, 6336)
            new_obs = torch.cat((old_obs.reshape(8, 66, 96), torch.randn(8, 66, 2)), -1).flatten(1)
            torch.testing.assert_close(source.actor_critic.act_inference(old_obs), target.actor_critic.act_inference(new_obs), rtol=1e-6, atol=1e-6)
            phase_columns = torch.cat([torch.tensor([96, 97])+98*i for i in range(5)])
            self.assertTrue((target.actor_critic.actor[0].weight[:, phase_columns] == 0).all())
            self.assertTrue((target.actor_critic.state_estimator[0].weight[:, phase_columns] == 0).all())
            for key, value in critic.items():
                torch.testing.assert_close(value, target.actor_critic.critic.state_dict()[key], atol=0, rtol=0)
            for key, value in discriminator.items():
                torch.testing.assert_close(value, target.discriminator.state_dict()[key], atol=0, rtol=0)
            self.assertFalse(target.optimizer.state or target.disc_optimizer.state or target.replay.size)
            self.assertEqual(target.current_learning_iteration, 0)
            self.assertIn("append_phase", target.training_origin["observation_mapping"])
            target.learn(1)
            self.assertTrue(target.optimizer.state and target.disc_optimizer.state)
            saved = Path(td)/"phase.pt"; target.save(saved)
            restored = cpu_runner(self.lib, phase=True); restored.load(saved)
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertEqual(restored.training_origin, target.training_origin)
            jit = torch.jit.script(target.actor_critic.get_policy_module())
            torch.testing.assert_close(jit(new_obs), target.actor_critic.act_inference(new_obs))
            with self.assertRaises(ValueError):
                cpu_runner(self.lib).load(saved)

    def test_incompatible_source_is_rejected_without_mutating_the_policy(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            source = cpu_runner(self.lib, source=True); source.learn(1)
            path = Path(td)/"source.pt"; source.save(path)
            state = torch.load(path, map_location="cpu")
            for kind in ("dt", "critic_shape", "nan"):
                bad = copy.deepcopy(state)
                if kind == "dt": bad["metadata"]["policy_dt"] = .01
                if kind == "critic_shape": bad["model_state_dict"]["critic.0.weight"] = torch.zeros(32, 102)
                if kind == "nan": bad["model_state_dict"]["actor.0.weight"][0, 0] = float("nan")
                torch.save(bad, path); target = cpu_runner(self.lib, phase=True)
                before = copy.deepcopy(target.actor_critic.state_dict())
                with self.assertRaises(ValueError): target.warm_start(path)
                for key, value in before.items():
                    torch.testing.assert_close(value, target.actor_critic.state_dict()[key], rtol=0, atol=0)

    def environment(self):
        class Parent:
            def _policy_frame(self): return self.core_frame
            def set_training_iteration(self, iteration):
                if iteration < 0: raise ValueError("negative iteration")
            def _post_physics_step_callback(self): self.callback_calls += 1
            def step(self, actions):
                self.commands[:, 0] = .4
                return self.obs_buf, self.privileged_obs_buf, None, None, None
            def compute_reward(self):
                self.reward_values = [getattr(self, "_reward_reference_"+key)() for key in ("pose", "velocity", "feet")]
                self.extras = {"training_metrics": {}}
        ns = configuration114("114")
        ns.update(torch=torch, math=math, X1AMPExp113Env=Parent)
        definitions("humanoid/envs/x1/x1_amp_exp114_common_env.py", ns)
        definitions("humanoid/envs/x1/x1_amp_exp114_env.py", ns)
        env = ns["X1AMPExp114Env"]();env.cfg=ns["X1AMPExp114Cfg"]()
        env.num_envs, env.dt, env.device = 3, .02, "cpu"
        env.dof_names = self.lib.dof_names
        clip = self.lib.names.index("F1_B1_stand_to_walk")
        env.standing_state = self.lib.state(torch.tensor([clip]), torch.tensor([11/50]))[0]
        env.reference_guide = PhaseReference(self.lib, env.standing_state[13:42], env.cfg.amp.reference_cycles)
        env.reference_foot_fk = URDFKinematics(URDF, env.dof_names, ["left_ankle_roll_link", "right_ankle_roll_link"])
        env.reference_foot_columns = list(range(68, 74))
        env.commands = torch.tensor([[0., 0., 0.], [.6, 0., 0.], [-.2, 0., 0.]])
        env.gait_phase = torch.tensor([.2, .3, .4]);env.previous_phase_command=torch.zeros(3)
        env.dof_pos, env.dof_vel = env.reference_guide.sample(env.commands[:, 0], env.gait_phase)
        env.core_frame=torch.randn(3,96);env.obs_buf=torch.randn(3,6468);env.privileged_obs_buf=torch.randn(3,101)
        env.obs_history=[torch.randn(3,98)];env.callback_calls=0
        env.root_states=torch.zeros(3,13);env.root_states[:,2]=.65;env.env_origins=torch.zeros(3,3)
        env.projected_gravity=torch.tensor([[0.,0.,-1.]]*3);env.reset_buf=torch.zeros(3,dtype=torch.bool);env.time_out_buf=env.reset_buf.clone()
        env.amp_history=torch.zeros(3,3,86);env.amp_history[:,-1,68:74]=env.reference_foot_fk.key_positions(env.dof_pos).flatten(-2)
        return env

    def test_actual_clock_and_buffer_transition_keep_latest_phase_observation_consistent(self):
        env=self.environment();phase=env.gait_phase.clone()
        env._policy_frame();env._policy_frame()
        torch.testing.assert_close(env.gait_phase,phase)
        env._post_physics_step_callback()
        expected=(phase+env.reference_guide.phase_rate(env.commands[:,0])*.02).remainder(1)
        expected[0]=phase[0]
        torch.testing.assert_close(env.gait_phase,expected)
        self.assertEqual(env.callback_calls,1)
        last_velocity=env.privileged_obs_buf[:,-3:].clone()
        env.step(torch.zeros(3,29))
        torch.testing.assert_close(env.obs_buf[:,-2:],env._phase_features())
        torch.testing.assert_close(env.obs_history[-1][:,-2:],env._phase_features())
        torch.testing.assert_close(env.privileged_obs_buf[:,96:98],env._phase_features())
        torch.testing.assert_close(env.privileged_obs_buf[:,-3:],last_velocity)
        torch.testing.assert_close(env._phase_features().norm(dim=-1),torch.ones(3))

    def test_actual_rewards_are_dense_and_gated_with_full_standing_constraints(self):
        env=self.environment();env.root_states[2,2]=.2
        for iteration, weight in ((0,.25),(500,.625),(1000,1.),(6000,1.)):
            env.set_training_iteration(iteration);env.compute_reward()
            for value in env.reward_values:
                torch.testing.assert_close(value,torch.tensor([0.,weight,0.]))
            self.assertTrue(all(torch.isfinite(v).all() for v in env.extras["training_metrics"].values()))


if __name__ == "__main__": unittest.main()
