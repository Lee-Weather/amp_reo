"""Bounded GPU preflight: short AMP training, exact checkpoint restore, resume."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import isaacgym  # Import before torch.
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def equal_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual), "Checkpoint tensor changed"
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            equal_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            equal_state(left, right)
    else:
        assert expected == actual


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_iterations", type=int, default=20)
    parser.add_argument("--resume_iterations", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, required=True)
    smoke, remaining = parser.parse_known_args()
    if min(smoke.smoke_iterations, smoke.resume_iterations) < 1:
        raise ValueError("Both training phases must have positive iteration counts")
    output = smoke.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    if args.task not in ("x1_amp_exp3", "x1_amp_exp14"):
        raise ValueError("This preflight validates x1_amp_exp3 or x1_amp_exp14")
    args.resume = False
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.seed is not None:
        cfg.seed = args.seed
    train_cfg.runner.resume = False
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
    initial_iteration = runner.current_learning_iteration
    if args.warm_start:
        assert initial_iteration == 6000
        expected_source = torch.load(args.warm_start, map_location=runner.device)
        equal_state(expected_source["model_state_dict"], runner.actor_critic.state_dict())
        equal_state(expected_source["discriminator_state_dict"], runner.discriminator.state_dict())
        assert not runner.optimizer.state and not runner.disc_optimizer.state
        assert runner.optimizer.param_groups[0]["lr"] == train_cfg.algorithm.learning_rate
        del expected_source
    assert env.device.startswith("cuda"), "Expected CUDA simulation"
    assert runner.device.startswith("cuda"), "Expected CUDA policy"
    assert env.num_envs == (args.num_envs or cfg.env.num_envs)
    assert env.motion_library.history_steps == 3
    assert runner.disc_cfg["style_reward_scale"] == 10.
    assert cfg.rewards.tracking_lin_sigma == .09 and cfg.rewards.tracking_sigma == .25
    records = []

    def watch(current, phase):
        original = current.update_policy
        def update(batch):
            assert torch.isfinite(batch["obs"]).all()
            assert torch.isfinite(batch["returns"]).all()
            assert torch.isfinite(batch["amp"]).all()
            torch.testing.assert_close(batch["style_rewards"], batch["style_scores"]*10*env.dt)
            metrics = original(batch)
            assert all(math.isfinite(value) for value in metrics.values())
            if args.warm_start and current.current_learning_iteration == initial_iteration:
                assert metrics["kl"] <= 1.5*current.alg_cfg["desired_kl"], "Warm-start first update overshot KL"
            assert (env.commands[:, 1:3] == 0).all()
            assert ((env.commands[:, 0] >= -.2) & (env.commands[:, 0] <= .65)).all()
            assert torch.isfinite(env.dof_pos).all() and torch.isfinite(env.dof_vel).all()
            target = env.pd_target_effective
            assert (target >= env.hard_dof_limits[:, 0]-1e-6).all()
            assert (target <= env.hard_dof_limits[:, 1]+1e-6).all()
            lr = current.optimizer.param_groups[0]["lr"]
            assert 1e-5 <= lr <= 1e-4
            records.append(dict(phase=phase, iteration=current.current_learning_iteration+1,
                                learning_rate=lr, task_reward=batch["task_rewards"].mean().item(),
                                style_reward=batch["style_rewards"].mean().item(),
                                style_score=batch["style_scores"].mean().item(), **metrics))
            return metrics
        current.update_policy = update

    watch(runner, "fresh")
    runner.learn(smoke.smoke_iterations)
    saved_lr = runner.optimizer.param_groups[0]["lr"]
    with tempfile.TemporaryDirectory(prefix="checkpoint_", dir=str(output)) as temporary:
        checkpoint = Path(temporary)/"model.pt"
        runner.save(str(checkpoint))
        expected = torch.load(str(checkpoint), map_location=runner.device)
        resumed, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
        resumed.load(str(checkpoint))
        equal_state(expected["model_state_dict"], resumed.actor_critic.state_dict())
        equal_state(expected["discriminator_state_dict"], resumed.discriminator.state_dict())
        equal_state(expected["optimizer_state_dict"], resumed.optimizer.state_dict())
        equal_state(expected["disc_optimizer_state_dict"], resumed.disc_optimizer.state_dict())
        assert resumed.current_learning_iteration == initial_iteration+smoke.smoke_iterations
        assert resumed.optimizer.param_groups[0]["lr"] == saved_lr
        watch(resumed, "resumed")
        resumed.learn(smoke.resume_iterations)
        assert resumed.current_learning_iteration == initial_iteration+smoke.smoke_iterations+smoke.resume_iterations
        policy = resumed.get_inference_policy(device=env.device)
        with torch.no_grad():
            actions = policy(env.get_observations())
        assert actions.shape == (env.num_envs, 29) and torch.isfinite(actions).all()
    report = dict(task=args.task, seed=cfg.seed, num_envs=env.num_envs, policy_dt=env.dt,
                  physics_dt=cfg.sim.dt, gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, initial_iteration=initial_iteration,
                  fresh_iterations=smoke.smoke_iterations,
                  resumed_iterations=smoke.resume_iterations, final_iteration=resumed.current_learning_iteration,
                  checkpoint_restore_exact=True, temporary_checkpoint_removed=True,
                  inference_shape=list(actions.shape), status="passed", updates=records)
    (output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("GPU_SMOKE_OK "+json.dumps({key:value for key,value in report.items() if key != "updates"}), flush=True)


if __name__ == "__main__":
    main()
