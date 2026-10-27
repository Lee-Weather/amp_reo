"""Deploy a fresh 201.5 workspace and warm-start AMP using a verified local model."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile


HOST = "robot@10.12.201.5"
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
RESOURCES = "/home/robot/czy/exp1/exp_20261026_151157/resources"


def remote(command, program=None):
    result = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, command],
                            input=program.encode() if program else None, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])
    return result.stdout.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.state.exists():
        raise ValueError("State already exists: inspect its process; never launch a duplicate")
    if args.iterations <= 0 or args.task != "x1_amp_"+args.experiment.replace(".", ""):
        raise ValueError("Expected a positive update count and matching x1_amp_exp1x task")
    import torch
    source_bytes = args.model.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != args.sha256:
        raise ValueError("Source model SHA256 mismatch")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["iter"])
    if not all(torch.isfinite(v).all() for key in ("model_state_dict", "discriminator_state_dict")
               for v in checkpoint[key].values()):
        raise ValueError("Source weights contain non-finite values")
    # Refuse to overlap any existing GPU compute workload.
    busy = remote("nvidia-smi --query-compute-apps=pid --format=csv,noheader").strip()
    if busy:
        raise RuntimeError("GPU already has compute processes: "+busy)
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    server = "/home/robot/czy/exp1/exp_"+stamp
    run_name = args.experiment+("_preflight" if args.preflight else "")
    source_path = "checkpoints/model_"+str(iteration)+".pt"
    command = ["python", "-u", "humanoid/scripts/train.py", "--task="+args.task, "--headless",
               "--num_envs=4096", "--seed=5", "--max_iterations="+str(args.iterations),
               "--run_name="+run_name, "--warm_start="+source_path, "--warm_start_sha256="+digest]
    code_digest = hashlib.sha256()
    code_files = sorted((root/"humanoid").rglob("*.py")) + [root/"setup.py"]
    with tempfile.TemporaryDirectory(prefix="amp_server_") as temp:
        archive = Path(temp)/"deployment.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in code_files:
                relative = path.relative_to(root).as_posix()
                code_digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
                tar.add(path, arcname=relative)
            tar.add(args.model, arcname=source_path)
        remote("mkdir "+shlex.quote(server))
        subprocess.run([SCP, "-o", "BatchMode=yes", str(archive), HOST+":"+server+"/deployment.tar.gz"],
                       check=True, timeout=120)
    script = "\n".join([
        "set -euo pipefail", "cd "+shlex.quote(server),
        "tar -xzf deployment.tar.gz", "ln -s "+shlex.quote(RESOURCES)+" resources",
        "source /home/robot/Anaconda/etc/profile.d/conda.sh", "conda activate F1",
        "export PYTHONPATH="+shlex.quote(server), "export OMP_NUM_THREADS=1",
        "pip install -e . --no-deps", shlex.join(command), "echo SERVER_TRAINING_OK", ""])
    state = dict(experiment=args.experiment, backend="ssh_server", host=HOST, server_dir=server,
                 source_iteration=iteration, source_checkpoint_sha256=digest,
                 source_model=str(args.model).replace("\\", "/"), target_iteration=iteration+args.iterations,
                 max_iterations=args.iterations, num_envs=4096, seed=5, task=args.task,
                 resume_mode="warm_start_fresh_optimizers", preflight=args.preflight,
                 branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                 commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 code_sha256=code_digest.hexdigest(), command=shlex.join(command),
                 log_file="train.log", success_marker="SERVER_TRAINING_OK",
                 resource_source=RESOURCES, phase="prepared", launch_pid=None)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    remote("cat > "+shlex.quote(server+"/run_train.sh"), script)
    # State is persisted before launch. If observation times out, inspect this
    # exact workspace and process before taking any further launch action.
    launch = ("nohup bash "+shlex.quote(server+"/run_train.sh")+" > "
              +shlex.quote(server+"/train.log")+" 2>&1 < /dev/null &")
    pid = remote("bash -c "+shlex.quote(launch+" echo $!" )).strip()
    if not pid.isdigit():
        raise RuntimeError("Inspect prepared workspace; launcher PID could not be read")
    state.update(launch_pid=pid, phase="launched", launched_at=datetime.now().astimezone().isoformat())
    args.state.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
