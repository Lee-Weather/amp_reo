"""Download a trusted training checkpoint and verify its saved-file receipt."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import torch

import flux_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    flux_ops.ACCOUNT = state["account"]
    iteration = args.iteration or state["target_iteration"]
    name = "model_{}.pt".format(iteration)
    log_path = args.state.with_suffix(".log")
    log = log_path.read_text(encoding="utf-8")
    receipts = [json.loads(value) for value in re.findall(r"CHECKPOINT_SAVED\s+(\{[^\n]+?\})", log)]
    receipts = [value for value in receipts if value["iteration"] == iteration]
    if not receipts:
        raise RuntimeError("Refresh task logs; no local save receipt is available for this iteration")
    expected = receipts[-1]
    rows = flux_ops.call("task", "model", "list", "--task-id", state["task_id"],
                         "--checkpoint", str(iteration), "--limit", "30")["data"]["rows"]
    matching = [row for row in rows if row["fileName"] == name and str(row["checkpoint"]) == str(iteration)]
    if not matching:
        raise RuntimeError("The requested checkpoint is not uploaded yet")
    model = max(matching, key=lambda row: int(row["id"]))
    output = args.output_dir.resolve()
    if flux_ops.ROOT not in output.parents:
        raise ValueError("Output must be inside the workspace")
    with tempfile.TemporaryDirectory(prefix="amp_verified_download_") as temporary:
        staged = Path(temporary)/name
        url = model["policUrlDown"]
        if not url.startswith("https://"):
            raise ValueError("Checkpoint download requires HTTPS")
        config = 'url = "'+url.replace('"', '%22')+'"\n'
        result = subprocess.run([
            "C:/Windows/System32/curl.exe", "-sS", "-L", "--ssl-no-revoke", "--fail",
            "--connect-timeout", "15", "--max-time", "120", "--proxy", "socks5h://10.12.201.122:39009",
            "--config", "-", "--output", str(staged)], input=config.encode(), capture_output=True)
        if result.returncode:
            raise RuntimeError("Checkpoint download failed, exit={}; signed URL withheld".format(result.returncode))
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        if digest != expected["sha256"] or staged.stat().st_size != expected["bytes"]:
            raise ValueError("Downloaded checkpoint differs from its saved-file receipt")
        checkpoint = torch.load(staged, map_location="cpu", weights_only=False)
        reference = torch.load(args.reference_model, map_location="cpu", weights_only=False)
        assert checkpoint["iter"] == iteration
        assert checkpoint["metadata"] == reference["metadata"]
        assert checkpoint["training_origin"]["source_sha256"] == state["source_checkpoint_sha256"]
        assert checkpoint["optimizer_state_dict"]["state"] and checkpoint["disc_optimizer_state_dict"]["state"]
        output.mkdir(parents=True, exist_ok=True)
        destination = output/name
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise FileExistsError("Preserve an existing, different model: "+str(destination))
        shutil.copy2(staged, destination)
    validation = {"iteration": iteration, "model_id": model["id"], "bytes": expected["bytes"],
                  "sha256": digest, "path": destination.relative_to(flux_ops.ROOT).as_posix(),
                  "robot_metadata_matches_reference": True, "optimizer_states_present": True,
                  "verified_at": datetime.now().astimezone().isoformat()}
    state.setdefault("verified_checkpoints", {})[str(iteration)] = validation
    args.state.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print("CHECKPOINT_DOWNLOAD_VERIFIED "+json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
