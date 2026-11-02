"""Phase-free task uses actual transition feedback without changing legacy DH inputs."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from humanoid.amp.reference_distance import ReferenceTransition
from humanoid.amp.stationary import CommandMotionSampler
from reference_task_fixtures import configuration114, cpu_runner
from test_amp import library
from test_exp3 import definitions


class Exp114tTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.lib=library()

    def test_configuration_keeps_interface_and_matches_shared_budget_and_source(self):
        ns=configuration114("114t");cfg,ppo=ns["X1AMPExp114tCfg"],ns["X1AMPExp114tCfgPPO"]
        old=ns["X1AMPExp113Cfg"]
        for key in ("env","control","commands","asset","amp","sim","noise","domain_rand"):
            self.assertIs(getattr(cfg,key),getattr(old,key))
        self.assertEqual((cfg.env.num_observations,cfg.env.num_privileged_obs,cfg.env.num_actions),(6336,99,29))
        self.assertEqual((cfg.rewards.curriculum_initial_stand_weight,cfg.rewards.scales.stand_pose,cfg.rewards.scales.raw_target_violation),(1.,-1.2,-2.))
        self.assertEqual(cfg.rewards.scales.reference_transition,4.)
        self.assertEqual((ppo.runner.max_iterations,ppo.runner.num_steps_per_env,ppo.algorithm.num_learning_epochs,ppo.algorithm.num_mini_batches,ppo.discriminator.num_updates),(6000,24,2,4,4))
        self.assertEqual(ppo.runner.warm_start_mode,"policy_only")

    def test_actual_reward_cache_zero_command_falls_and_static_credit(self):
        class Parent:
            def set_training_iteration(self,iteration): pass
            def compute_reward(self):
                self.reward_values=self._reward_reference_transition()
                self.extras={"training_metrics":{}}
        ns=configuration114("114t");ns.update(torch=torch,X1AMPExp113Env=Parent)
        definitions("humanoid/envs/x1/x1_amp_exp114_common_env.py",ns)
        definitions("humanoid/envs/x1/x1_amp_exp114t_env.py",ns)
        env=ns["X1AMPExp114tEnv"]();env.cfg=ns["X1AMPExp114tCfg"]();env.dof_names=self.lib.dof_names
        sampler=CommandMotionSampler(self.lib,exclude_quiet_for_moving=True,relative_speed_tolerance=.5)
        env.reference_guide=ReferenceTransition(sampler)
        env.commands=torch.tensor([[0.,0.,0.],[.6,0.,0.],[-.2,0.,0.]])
        ids,times=sampler.sample_matching(env.commands);states=self.lib.state(ids,times)
        env.amp_history=self.lib.history(ids,times);env.dof_pos=states[:,13:42]
        env.standing_state=states[0];env.root_states=states[:,:13].clone();env.root_states[:,2]=.65;env.root_states[2,2]=.2
        env.env_origins=torch.zeros(3,3);env.projected_gravity=torch.tensor([[0.,0.,-1.]]*3)
        env.reset_buf=torch.zeros(3,dtype=torch.bool);env.time_out_buf=env.reset_buf.clone();env.common_step_counter=1;env._reference_cache_step=-1
        with patch.object(env.reference_guide,"reward",wraps=env.reference_guide.reward) as count:
            env.compute_reward();env._reward_reference_transition()
            self.assertEqual(count.call_count,1)
            self.assertEqual(float(env.reward_values[0]),0.)
            self.assertGreater(float(env.reward_values[1]),0.)
            self.assertEqual(float(env.reward_values[2]),0.)
            env.amp_history[:]=env.amp_history[:,-1:].clone()
            env.amp_history[:,:,env.reference_guide.velocity_columns]=0
            env.common_step_counter+=1;env.compute_reward()
            self.assertEqual(count.call_count,2)
            torch.testing.assert_close(env.reward_values,torch.zeros(3),atol=1e-6,rtol=0)

    def test_strict_policy_transfer_still_trains_and_restores_full_state(self):
        source=cpu_runner(self.lib,source=True)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            source.learn(2);source.current_learning_iteration=6000
            path=Path(td)/"source.pt";source.save(path)
            target=cpu_runner(self.lib);target.warm_start(path)
            obs=torch.randn(4,6336)
            torch.testing.assert_close(source.actor_critic.act_inference(obs),target.actor_critic.act_inference(obs),atol=0,rtol=0)
            target.learn(1);path=Path(td)/"new.pt";target.save(path)
            restored=cpu_runner(self.lib);restored.load(path)
            self.assertTrue(restored.optimizer.state and restored.disc_optimizer.state)
            self.assertEqual(restored.current_learning_iteration,1)
            torch.testing.assert_close(restored.actor_critic.act_inference(obs),target.actor_critic.act_inference(obs),atol=0,rtol=0)


if __name__ == "__main__": unittest.main()
