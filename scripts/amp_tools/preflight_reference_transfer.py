"""CPU-only production-checkpoint transfer/update/export check (synthetic environment)."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]
from humanoid.amp.runner import AMPOnPolicyRunner
from reference_task_fixtures import configuration114
from test_amp import library
from test_exp110 import DHFakeEnv


def dictionary(value):
    if isinstance(value, (str, bool, int, float, list, tuple, dict)) or value is None:
        return copy.deepcopy(value)
    return {name: dictionary(getattr(value, name)) for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))}


def check(route, source_path):
    torch.set_num_threads(2)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    ns = configuration114(route)
    settings = ns["X1AMPExp"+route+"Cfg"]()
    cfg = dictionary(ns["X1AMPExp"+route+"CfgPPO"]())
    expected = cfg["runner"]["policy_transfer_source_sha256"]
    assert digest == expected and source["iter"] == 6000
    for section in ("control", "commands", "asset", "sim", "domain_rand", "noise"):
        configured = dictionary(getattr(settings, section))
        if section == "domain_rand":
            for interval in ("ext_force_interval", "push_interval"):
                configured[interval] = math.ceil(configured[interval+"_s"]/source["metadata"]["policy_dt"])
        assert configured == source["config"][section], section
    # Gym's imported DOF traversal differs from the XML declaration order.
    lib = library(source["metadata"]["dof_names"])

    def environment(width, metadata):
        env = DHFakeEnv(lib)
        env.dof_names = list(source["metadata"]["dof_names"])
        env.cfg.env.num_single_obs = width
        env.cfg.control = settings.control
        env.num_obs, env.num_privileged_obs = width*66, width+3
        env.obs, env.critic = torch.zeros(4, env.num_obs), torch.zeros(4, env.num_privileged_obs)
        env.dt = source["metadata"]["policy_dt"]
        env.default_dof_pos = torch.tensor(source["metadata"]["default_dof_pos"])[None]
        env.checkpoint_control_metadata = lambda: copy.deepcopy(source["metadata"]["control"])
        env.checkpoint_training_metadata = lambda: copy.deepcopy(metadata)
        return env

    source_runner = AMPOnPolicyRunner(environment(96, source["metadata"]["training"]), source["config"])
    with contextlib.redirect_stdout(io.StringIO()):
        source_runner.load(str(source_path))
    target = AMPOnPolicyRunner(environment(settings.env.num_single_obs, {"cpu_fixture_route": route}), cfg)
    critic = copy.deepcopy(target.actor_critic.critic.state_dict())
    discriminator = copy.deepcopy(target.discriminator.state_dict())
    with contextlib.redirect_stdout(io.StringIO()):
        target.warm_start(str(source_path), digest)
    assert target.current_learning_iteration == 0
    assert not target.optimizer.state and not target.disc_optimizer.state and target.replay.size == 0
    for key, value in critic.items():
        assert torch.equal(target.actor_critic.critic.state_dict()[key], value)
    for key, value in discriminator.items():
        assert torch.equal(target.discriminator.state_dict()[key], value)
    maximum = 0.
    with torch.no_grad():
        for _ in range(16):
            old = torch.randn(256, 6336)*.1
            new = old if route == "114t" else torch.cat((old.view(-1, 66, 96), torch.randn(256, 66, 2)), -1).flatten(1)
            a, b = source_runner.actor_critic.act_inference(old), target.actor_critic.act_inference(new)
            assert torch.isfinite(b).all()
            maximum = max(maximum, float((a-b).abs().max()))
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    # This is a real PPO/AMP update with the production networks; physics is synthetic.
    batch = target.collect_rollout()
    metrics = target.update_policy(batch)
    metrics.update(target.update_discriminator(batch["amp"]))
    assert all(math.isfinite(v) for v in metrics.values())
    assert target.optimizer.state and target.disc_optimizer.state
    target.current_learning_iteration = 1
    with tempfile.TemporaryDirectory(prefix="amp-reference-cpu-") as tmp:
        checkpoint = str(Path(tmp)/"model_1.pt")
        with contextlib.redirect_stdout(io.StringIO()):
            target.save(checkpoint)
            restored = AMPOnPolicyRunner(environment(settings.env.num_single_obs, {"cpu_fixture_route": route}), cfg)
            restored.load(checkpoint)
        for key, value in target.actor_critic.state_dict().items():
            assert torch.equal(restored.actor_critic.state_dict()[key], value)
        for p, q in zip(target.actor_critic.parameters(), restored.actor_critic.parameters()):
            for key, value in target.optimizer.state[p].items():
                assert torch.equal(value, restored.optimizer.state[q][key])
        exported = torch.jit.script(copy.deepcopy(target.actor_critic.get_policy_module()).cpu().eval())
        error = float((exported(batch["obs"])-target.actor_critic.act_inference(batch["obs"])).abs().max())
        assert error <= 1e-6
    return {"verified": True, "scope": "CPU synthetic physics; production source, networks, optimizer update and export",
            "source_sha256": digest, "source_iteration": 6000, "new_initial_iteration": 0,
            "route": route, "num_observations": target.env.num_obs, "num_actions": 29,
            "finite_inference_samples": 4096, "inference_batch_size": 256,
            "transfer_inference_max_error": maximum, "fresh_critic_discriminator_optimizers_replay": True,
            "ppo_amp_update_metrics": metrics, "checkpoint_resume_exact": True,
            "complete_jit_max_error": error,
            "formal_gpu_physics_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("114", "114t"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.route, args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
