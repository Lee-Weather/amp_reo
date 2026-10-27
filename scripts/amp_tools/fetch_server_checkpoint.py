"""Collect an SSH recovery checkpoint and verify model and optimizer continuity."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]


def optimizer_steps(state):
    values = [float(value["step"]) for value in state["state"].values()]
    if not values:
        raise ValueError("Empty optimizer state")
    return {"minimum": min(values), "maximum": max(values), "state_count": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["backend"] != "ssh_server" or state["host"] != "robot@10.12.201.5":
        raise ValueError("Use the recorded 201.5 recovery state")
    iteration = args.iteration or state["target_iteration"]
    if iteration == state["target_iteration"] and state["phase"] != "completed_training":
        raise RuntimeError("Refresh server status and wait for the recorded process to finish")
    receipts = [r for r in state["status"]["saved_checkpoints"] if r["iteration"] == iteration]
    if not receipts:
        raise RuntimeError("Refresh server status; no saved checkpoint receipt for this iteration")
    receipt = receipts[-1]
    if not PurePosixPath(receipt["path"]).is_relative_to(PurePosixPath(state["server_dir"])):
        raise ValueError("Checkpoint is outside the recorded server run")
    if hashlib.sha256(args.reference_model.read_bytes()).hexdigest() != state["source_checkpoint_sha256"]:
        raise ValueError("Use the exact recovery source checkpoint as reference")
    source = torch.load(args.reference_model, map_location="cpu", weights_only=False)
    output = args.output_dir.resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be within the workspace")
    with tempfile.TemporaryDirectory(prefix="x1_verified_server_checkpoint_") as td:
        staged = Path(td)/"model_{}.pt".format(iteration)
        subprocess.run(["C:/Windows/System32/OpenSSH/scp.exe", "-o", "BatchMode=yes",
                        state["host"]+":"+receipt["path"], str(staged)], check=True)
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        if digest != receipt["sha256"] or staged.stat().st_size != receipt["bytes"]:
            raise ValueError("Server checkpoint differs from its save receipt")
        checkpoint = torch.load(staged, map_location="cpu", weights_only=False)
        assert checkpoint["iter"] == iteration > source["iter"]
        assert checkpoint["metadata"] == source["metadata"]
        assert checkpoint["training_origin"] == source["training_origin"]
        assert all(torch.isfinite(value).all() for group in ("model_state_dict", "discriminator_state_dict")
                   for value in checkpoint[group].values())
        counters = {}
        for key in ("optimizer_state_dict", "disc_optimizer_state_dict"):
            counters[key] = optimizer_steps(checkpoint[key])
            previous = optimizer_steps(source[key])
            assert counters[key]["state_count"] == previous["state_count"]
            assert counters[key]["minimum"] > previous["maximum"]
        assert checkpoint["discriminator_state_dict"]["normalizer.count"] > source["discriminator_state_dict"]["normalizer.count"]
        output.mkdir(parents=True, exist_ok=True)
        destination = output/staged.name
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise FileExistsError("Preserve the existing different checkpoint")
        shutil.copy2(staged, destination)
    validation = {"iteration": iteration, "sha256": digest, "bytes": receipt["bytes"],
                  "path": destination.relative_to(ROOT).as_posix(), "metadata_matches_source": True,
                  "weights_finite": True, "original_training_origin_retained": True,
                  "optimizer_counters": counters, "normalizer_count_increased": True,
                  "verified_at": datetime.now().astimezone().isoformat()}
    state.setdefault("verified_checkpoints", {})[str(iteration)] = validation
    args.state.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print("SERVER_CHECKPOINT_VERIFIED "+json.dumps(validation))


if __name__ == "__main__":
    main()
