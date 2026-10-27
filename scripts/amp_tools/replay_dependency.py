"""Wait for a recorded recovery to finish and copy its verified final checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def prepare_checkpoint(training_root, process_alive, iteration, destination,
                       log_file="train_recovery.log", success_marker="SERVER_RECOVERY_OK"):
    root, destination = Path(training_root).resolve(), Path(destination)
    log = (root/log_file).read_text(errors="replace")
    if process_alive:
        return None
    if success_marker not in log:
        raise RuntimeError("Training stopped without its completion marker; replay not started")
    receipts = [json.loads(line.split("CHECKPOINT_SAVED ", 1)[1]) for line in log.splitlines()
                if line.startswith("CHECKPOINT_SAVED ")]
    receipt = next((r for r in reversed(receipts) if r["iteration"] == iteration), None)
    if receipt is None:
        raise RuntimeError("Missing final checkpoint receipt; replay not started")
    source = Path(receipt["path"]).resolve()
    source.relative_to(root)  # Refuse a path outside the recorded training run.
    if source.stat().st_size != receipt["bytes"] or hashlib.sha256(source.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("Final checkpoint differs from its save receipt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != receipt["sha256"]:
        raise FileExistsError("Preserve an existing different replay checkpoint")
    shutil.copy2(source, destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == receipt["sha256"]
    return {"iteration": iteration, "sha256": receipt["sha256"], "bytes": receipt["bytes"],
            "source_training_dir": str(root), "source_checkpoint": str(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--log-file", default="train_recovery.log")
    parser.add_argument("--success-marker", default="SERVER_RECOVERY_OK")
    parser.add_argument("--launcher-script", default="run_recovery.sh")
    parser.add_argument("--wait-gpu", action="store_true")
    args = parser.parse_args()
    print("WAITING_FOR_RECORDED_TRAINING", flush=True)
    while True:
        process = Path("/proc/{}/stat".format(args.pid))
        try:
            command = process.with_name("cmdline").read_bytes()
            alive = process.read_text().split()[2] != "Z" and str(args.training_root/args.launcher_script).encode() in command
        except FileNotFoundError:
            alive = False
        result = prepare_checkpoint(args.training_root, alive, args.iteration, args.destination,
                                    args.log_file, args.success_marker)
        if result is not None:
            Path("prepared_checkpoint.json").write_text(json.dumps(result, indent=2)+"\n")
            print("REPLAY_CHECKPOINT_READY "+json.dumps(result), flush=True)
            if args.wait_gpu:
                print("WAITING_FOR_IDLE_GPU", flush=True)
                while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                                "--format=csv,noheader"]).strip():
                    time.sleep(5)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
