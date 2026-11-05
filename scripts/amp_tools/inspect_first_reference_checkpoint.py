"""Inspect a verified uploaded first model in memory; do not retain intermediate PT files."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flux_ops as flux
from humanoid.amp.dh_models import ActorCriticDHAMP
from humanoid.amp.models import Discriminator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--via-server", action="store_true")
    parser.add_argument("--via-proxy", action="store_true", help="Download into local memory through the configured Git/Flux SOCKS proxy")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    directory = ROOT/"czy/exp1/records"/(state["experiment"]+"_evaluation")
    evidence = json.loads((directory/"first_checkpoint_verification.json").read_text())
    assert evidence["verified"] and evidence["first_iteration"] == 100
    assert evidence['task_id'] == state['task_id'] and evidence['commit'] == state['commit']
    flux.ACCOUNT = state["account"]
    rows = flux.call("task", "model", "list", "--task-id", state["task_id"], "--checkpoint", "100", "--limit", "30")["data"]["rows"]
    row = next(r for r in rows if r["fileName"] == "model_100.pt" and str(r["checkpoint"]) == "100")
    assert row["policUrlDown"].startswith("https://")
    try:
        if args.via_proxy:
            from private_training_input import request
            status, _, raw = request(row['policUrlDown'])
            assert status == 200
        elif args.via_server:
            program = ("import urllib.request,sys\ntry:\n with urllib.request.urlopen("+repr(row["policUrlDown"])+",timeout=90) as r:\n  b=r.read(32*1024*1024)\n sys.stdout.buffer.write(b)\nexcept Exception:\n sys.exit(1)\n")
            remote = subprocess.run(["C:/Windows/System32/OpenSSH/ssh.exe", "-o", "BatchMode=yes",
                "robot@10.12.201.5", "python3 -"], input=program.encode(), capture_output=True, timeout=120, check=True)
            raw = remote.stdout
        else:
            with urllib.request.urlopen(row["policUrlDown"], timeout=90) as response:
                raw = response.read(32*1024*1024)
    except Exception:
        raise RuntimeError("First-checkpoint HTTPS download failed; signed URL withheld") from None
    receipt = evidence["save_receipt"]
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == receipt["bytes"] and digest == receipt["sha256"]
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    source = torch.load(ROOT/state["source_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["iter"] == 100
    origin = checkpoint["training_origin"]
    assert origin["source_sha256"] == state["source_checkpoint_sha256"] and origin["source_iteration"] == state['source_iteration'] and origin["new_iteration"] == 0
    assert origin["mode"] == "policy_transfer_fresh_critic_amp_optimizers"
    metadata, cfg = checkpoint["metadata"], checkpoint["config"]
    assert metadata["control"] == source["metadata"]["control"]
    assert metadata["dof_names"] == source["metadata"]["dof_names"]
    assert metadata["policy_dt"] == source["metadata"]["policy_dt"]
    width = state['observation_dim']//66
    assert width in (96, 98)
    assert metadata["num_obs"] == 66*width and metadata["policy"]["num_single_obs"] == width
    assert cfg["env"]["num_envs"] == 4096 and cfg["seed"] == 5
    assert (cfg["runner"]["num_steps_per_env"], cfg["algorithm"]["num_learning_epochs"], cfg["algorithm"]["num_mini_batches"], cfg["discriminator"]["num_updates"]) == (24, 2, 4, 4)
    assert cfg["rewards"]["curriculum_initial_stand_weight"] == 1.
    assert cfg["rewards"]["scales"]["stand_pose"] == -1.2 and cfg["rewards"]["scales"]["raw_target_violation"] == -2.
    training = metadata["training"]
    assert training["reference_initial_weight"] == .25
    stage15 = state['experiment'] in ('exp1.15', 'exp1.15t')
    assert training["reference_warmup_iterations"] == (500 if stage15 else 1000)
    if stage15:
        assert training['curriculum'] == 'normalized_forward_guidance_v1'
        assert training['amp_style_curriculum'] == [500, 1500]
        assert training['command_events'] == {'forward_vx':.3, 'forward_probability':.8, 'zero_probability':.2}
        assert cfg['runner']['max_iterations'] == 1000 and cfg['commands']['backward_probability'] == 0.
        if state.get('teacher_sha256'):
            assert origin['teacher']['source_sha256'] == state['teacher_sha256']
            assert origin['teacher']['source_iteration'] == 11250
    for name, expected in training["reference_source_sha256"].items():
        path = ROOT/"resources/motions/slow_walk_amp_v1.2/normalized"/(name+".pkl")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    policy = copy.deepcopy(cfg["policy"])
    motion = policy["motion_exploration"]
    motion["indices"] = [metadata["dof_names"].index(n) for n in motion.pop("joint_names")]
    motion["command_index"] = 66*width-width
    motion["command_threshold"] = motion.pop("speed_threshold")*cfg["normalization"]["obs_scales"]["lin_vel"]
    model = ActorCriticDHAMP(66*width, width+3, 29, 5*width, width, **policy)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    discriminator = Discriminator(86, 3, cfg["discriminator"]["hidden_dims"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=True)
    for group in (model.state_dict(), discriminator.state_dict()):
        assert all(torch.isfinite(v).all() for v in group.values())
    counts = {}
    for name in ("optimizer_state_dict", "disc_optimizer_state_dict"):
        values = checkpoint[name]["state"]
        assert values
        assert all(torch.isfinite(v).all() for entry in values.values() for v in entry.values() if isinstance(v, torch.Tensor))
        steps = [float(entry["step"]) for entry in values.values()]
        counts[name] = {"parameter_states": len(values), "minimum_step": min(steps), "maximum_step": max(steps)}
    assert 100 <= counts["optimizer_state_dict"]["minimum_step"] <= counts["optimizer_state_dict"]["maximum_step"] <= 800
    assert counts["disc_optimizer_state_dict"]["minimum_step"] == counts["disc_optimizer_state_dict"]["maximum_step"] == 400
    normalizer_count = float(discriminator.normalizer.count)
    expected_count = 100*cfg["discriminator"]["num_updates"]*cfg["discriminator"]["batch_size"]*2*metadata["history_steps"]
    assert abs(normalizer_count-expected_count) < 1.
    torch.set_num_threads(2)
    with torch.no_grad():
        obs = torch.randn(4, 66*width)*.1
        exported = torch.jit.script(model.get_policy_module().eval())
        action = exported(obs)
        assert action.shape == (4, 29) and torch.isfinite(action).all()
        torch.testing.assert_close(action, model.act_inference(obs), atol=1e-6, rtol=1e-6)
    result = {"verified": True, "task_id": state["task_id"], "commit": state["commit"], "iteration": 100,
        "bytes": len(raw), "sha256": digest, "observation_dim": 66*width, "action_dim": 29,
        "actual_envs_seed_rollout_ppo_D": [4096, 5, 24, 2, 4, 4], "training_origin": origin,
        "control_unchanged": True, "full_standing_constraints": True, "reference_source_hashes_match": True,
        "reference_route": training["reference_route"], "optimizers": counts,
        "normalizer_count": normalizer_count, "fresh_discriminator_update_continuity": True,
        "all_parameters_and_optimizer_states_finite": True, "complete_policy_jit_matches": True}
    (directory/"first_checkpoint_inspection.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
