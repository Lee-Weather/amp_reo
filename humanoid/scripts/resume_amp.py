"""Continue a stopped cloud AMP run with its policy, discriminator and Adam state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def restore_continuation(runner, path, digest, source_iteration, target_iteration, iterations):
    import torch
    if (runner.current_learning_iteration != 0 or runner.training_origin is not None
            or runner.optimizer.state or runner.disc_optimizer.state or runner.replay.size):
        raise ValueError("Recovery requires a fresh runner before exact checkpoint restoration")
    if source_iteration < 0 or iterations <= 0 or source_iteration+iterations != target_iteration:
        raise ValueError("Recovery must finish the original target iteration")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("Recovery input SHA256 mismatch")
    state = torch.load(path, map_location=runner.device)
    if state["iter"] != source_iteration:
        raise ValueError("Recovery input iteration mismatch")
    if runner.cfg.get("require_fresh_initialization", False) and state.get("training_origin") is not None:
        raise ValueError("Scratch control cannot recover a transferred policy")
    for key in ("optimizer_state_dict", "disc_optimizer_state_dict"):
        if not state[key]["state"]:
            raise ValueError("Recovery requires both trained Adam states")
    runner.load(str(path), load_optimizer=True)
    steps = {}
    for label, optimizer in (("policy", runner.optimizer), ("discriminator", runner.disc_optimizer)):
        values = [float(v["step"]) for v in optimizer.state.values()]
        steps[label] = {"minimum": min(values), "maximum": max(values), "parameters": len(values)}
    return {"source_sha256": digest, "source_iteration": source_iteration,
            "target_iteration": target_iteration, "additional_updates": iterations,
            "training_origin": runner.training_origin, "optimizer_steps": steps,
            "learning_rate": runner.optimizer.param_groups[0]["lr"],
            "normalizer_count": float(runner.discriminator.normalizer.count),
            "replay_size": runner.replay.size,
            "limits": "Checkpoint does not store simulator/RNG/replay/episode-buffer state; these restart. Policy, critic, D, normalizer, both Adam states and absolute iteration are restored."}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recovery_model", type=Path, required=True)
    parser.add_argument("--recovery_sha256", required=True)
    parser.add_argument("--recovery_url_b64")
    parser.add_argument("--source_iteration", type=int, required=True)
    parser.add_argument("--target_iteration", type=int, required=True)
    recovery, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    # Isaac Gym must load before Torch on the official runtime.
    from humanoid.envs import task_registry
    from humanoid.utils import get_args
    from humanoid.amp.checkpoint import prepare_warm_start
    from torch.utils.tensorboard import SummaryWriter
    args = get_args()
    if args.resume or args.warm_start or args.checkpoint_dir or args.seed != 5:
        raise ValueError("Use seed5 and the explicit recovery input, without legacy resume/warm_start/output flags")
    if args.max_iterations is None or recovery.source_iteration+args.max_iterations != recovery.target_iteration:
        raise ValueError("Recovery update count must complete the original training budget")
    path = recovery.recovery_model.resolve()
    # Reuse only the verified download/shared-directory preparation, never
    # warm_start(): it would discard the saved optimizer states.
    prepare_warm_start(SimpleNamespace(checkpoint_dir=str(path.parent), warm_start=str(path),
                       warm_start_sha256=recovery.recovery_sha256,
                       warm_start_url_b64=recovery.recovery_url_b64, resume=False))
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, _, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=None)
    receipt = restore_continuation(runner, path, recovery.recovery_sha256,
                                  recovery.source_iteration, recovery.target_iteration, args.max_iterations)
    runner.log_dir = str(path.parent)
    runner.writer = SummaryWriter(runner.log_dir)
    (path.parent/"config.json").write_text(json.dumps(runner.config, indent=2)+"\n", encoding="utf-8")
    print("CLOUD_RECOVERY_RESTORED "+json.dumps(receipt), flush=True)
    runner.learn(args.max_iterations, init_at_random_ep_len=False)
    runner.writer.close()
    print("CLOUD_RECOVERY_OK "+json.dumps({"iteration": runner.current_learning_iteration}), flush=True)


if __name__ == "__main__":
    main()
