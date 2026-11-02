"""Deploy the current AMP play code, keeping temporary files out of analysis."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
HOST = "robot@10.12.201.5"


def gpu_workloads(output):
    """Allow the observed idle NoMachine desktop, but block other CUDA users."""
    active, desktop = [], []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Unexpected GPU process query; inspect before replay")
        pid, name, memory = (field.strip() for field in row)
        process = {"pid": int(pid), "process_name": name, "memory_mib": int(memory)}
        if name == "/usr/NX/bin/nxnode.bin" and process["memory_mib"] <= 512:
            desktop.append(process)
        else:
            active.append(process)
    return active, desktop


def source_files(commit=None, root=None):
    """Read an immutable training revision instead of later workspace edits."""
    root = ROOT if root is None else Path(root)
    if commit is None:
        paths = sorted((root/"humanoid").rglob("*.py"))+sorted((root/'humanoid/amp/reference_data').glob('*.json'))+[root/"setup.py"]
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in paths}, None
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("code-commit must be a full verified training commit SHA")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    revision = git("rev-parse", "--verify", commit+"^{commit}").decode().strip()
    names = git("ls-tree", "-r", "--name-only", "-z", revision, "--", "humanoid", "setup.py").decode().split("\0")
    names = sorted(n for n in names if n == "setup.py" or n.startswith("humanoid/") and n.endswith(".py")
        or n.startswith('humanoid/amp/reference_data/') and n.endswith('.json'))
    if "humanoid/scripts/play_amp.py" not in names or "setup.py" not in names:
        raise ValueError("Training revision does not contain the AMP replay entry point")
    return {name: git("show", revision+":"+name) for name in names}, revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resource_source", required=True)
    parser.add_argument("--task", default="x1_amp")
    parser.add_argument("--code-commit", help="Exact Git commit used by the completed cloud training")
    parser.add_argument("--training-task-id", default="TASK_20261022_177")
    parser.add_argument("--record_dir", type=Path)
    parser.add_argument("--extra_seeds", type=int, nargs="*", default=[])
    parser.add_argument("--check_rsi", action="store_true")
    parser.add_argument("--primary-vx", type=float, help="Record the main 60-second video at one requested vx instead of the historical six-stage profile")
    parser.add_argument("--check_speeds", type=float, nargs="*", default=[],
                        help="Additional constant-vx checks with standing reset and seed=5")
    parser.add_argument("--check_rsi_speeds", type=float, nargs="*", default=[],
                        help="Additional constant-vx checks with RSI reset and seed=5")
    parser.add_argument("--speed_check_steps", type=int, default=1000,
                        help="Steps per constant-speed check (default: 20 seconds at 50 Hz)")
    args = parser.parse_args()
    if args.speed_check_steps < 1000:
        raise ValueError("Constant-speed checks require at least 1000 steps")
    if any(not math.isfinite(vx) or not -.2 <= vx <= .65
           for vx in args.check_speeds+args.check_rsi_speeds+([] if args.primary_vx is None else [args.primary_vx])):
        raise ValueError("Speed checks must stay within the trained vx range [-0.2, 0.65]")
    if not re.fullmatch(r"exp[1-9][0-9]*(?:\.[0-9]+)?t?|exp_[0-9]+(?:\.[0-9]+)?", args.experiment):
        raise ValueError("Use an experiment name such as exp1.4 (or a historical exp_N name)")
    if not re.fullmatch(r"x1_amp(?:_exp[0-9]+t?)?", args.task):
        raise ValueError("Use a registered X1 AMP task")
    if any(seed != 5 for seed in args.extra_seeds):
        raise ValueError("Current experiments use seed=5 only")
    model = args.model.resolve()
    import torch
    iteration = int(torch.load(model, map_location="cpu", weights_only=False)["iter"])
    if args.task=='x1_amp_exp20':
        if args.primary_vx is None: args.primary_vx = .5953696966
        if args.check_rsi or args.check_rsi_speeds:
            raise ValueError('exp2.0 replay is standing-only')
        if any(abs(v)>1e-7 and abs(v-.5953696966)>.002 for v in args.check_speeds+[args.primary_vx]):
            raise ValueError('exp2.0 replay supports only zero and its native forward speed')
        args.primary_vx = .5953696966 if abs(args.primary_vx)>1e-7 else 0.
        args.check_speeds = [.5953696966 if abs(v)>1e-7 else 0. for v in args.check_speeds]
    if iteration < 0:
        raise ValueError("Checkpoint iteration must be nonnegative")
    model_name = "model_{}.pt".format(iteration)
    artifacts = args.record_dir.resolve() if args.record_dir else ROOT / "czy/exp1/records" / (args.experiment+"_evaluation")
    artifacts.mkdir(parents=True, exist_ok=True)
    state_file = artifacts / "replay_job.json"
    if state_file.exists():
        raise RuntimeError("An existing replay job is recorded; inspect its status before starting another")
    processes = subprocess.run([SSH, "-o", "BatchMode=yes", HOST,
                                "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=30).stdout
    busy, desktop = gpu_workloads(processes)
    if busy:
        raise RuntimeError("GPU already has compute workloads: "+json.dumps(busy))
    now = datetime.now(timezone(timedelta(hours=8)))
    remote = "/home/robot/czy/exp1/" + now.strftime("exp_%Y%m%d_%H%M%S")
    code, revision = source_files(args.code_commit)
    code_hash = hashlib.sha256()
    for name, content in sorted(code.items()):
        code_hash.update(name.encode())
        code_hash.update(content)
    resources = [path for folder in (ROOT/"resources/robots/x1_29",
                                     ROOT/"resources/motions/slow_walk_amp_v1.2")
                 for path in folder.rglob("*") if path.is_file()]
    if not resources:
        raise ValueError("Local replay resources are missing")
    resource_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(resources)}
    primary_output = "logs/replay/standing_seed5"
    checks = [{"seed": seed, "reset_mode": "standing", "output": "logs/replay/standing_seed"+str(seed)}
              for seed in args.extra_seeds]
    if args.check_rsi:
        checks.append({"seed": 5, "reset_mode": "rsi", "output": "logs/replay/rsi_seed5"})
    for index, vx in enumerate(args.check_speeds):
        checks.append({"seed": 5, "reset_mode": "standing", "vx": vx,
                       "steps": args.speed_check_steps,
                       "output": "logs/replay/speed_check_{}".format(index)})
    for index, vx in enumerate(args.check_rsi_speeds):
        checks.append({"seed": 5, "reset_mode": "rsi", "vx": vx,
                       "steps": args.speed_check_steps,
                       "output": "logs/replay/rsi_speed_check_{}".format(index)})
    state = {"experiment": args.experiment, "task_id": args.training_task_id, "task": args.task, "server_dir": remote,
             "checkpoint_iteration": iteration, "checkpoint_name": model_name,
             "created_at": now.isoformat(), "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
             "idle_desktop_processes": desktop,
             "code_sha256": code_hash.hexdigest(), "training_commit": revision,
             "code_hash_definition": ("SHA256 of sorted humanoid/*.py and humanoid/amp/reference_data/*.json plus setup.py path and contents"
                if any(n.endswith('.json') for n in code) else "SHA256 of sorted humanoid/*.py plus setup.py path and contents"),
             "source_file_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in code.items()},
             "resource_file_sha256": resource_hashes,
             "profile": [0, .4, .6, 0, .4, 0] if args.primary_vx is None else [args.primary_vx],
             "stage_seconds": 10 if args.primary_vx is None else 60,
             "steps": 3000, "video": "1920x1080@25fps",
             "primary_output": primary_output, "checks": checks}
    def run(command):
        subprocess.run(command, check=True)
    with tempfile.TemporaryDirectory(prefix="x1_amp_replay_") as temp:
        archive = Path(temp) / "code_model.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, content in code.items():
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(content), 0o644
                stream.addfile(entry, io.BytesIO(content))
            stream.add(model, arcname="checkpoints/"+model_name)
        script = Path(temp) / "run_replay.sh"
        expected = dict(state["source_file_sha256"], **resource_hashes)
        expected["checkpoints/"+model_name] = state["checkpoint_sha256"]
        verification = [
            "python3 - <<'VERIFY_DEPLOYMENT'",
            "import hashlib,json,pathlib",
            "expected="+repr(expected),
            "actual={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in expected}",
            "assert actual==expected, 'Replay deployment differs from training code, model or local resources'",
            "result=dict(verified=True,training_commit="+repr(revision)+",file_sha256=actual)",
            "pathlib.Path('deployment_verification.json').write_text(json.dumps(result,indent=2)+'\\n')",
            "print('DEPLOYMENT_VERIFIED',len(actual),flush=True)",
            "VERIFY_DEPLOYMENT"]
        base_command = ("python -u humanoid/scripts/play_amp.py --task="+args.task+
                        " --headless --num_envs=1 --checkpoint_path=checkpoints/"+model_name)
        primary_speed = "" if args.primary_vx is None else " --vx="+str(args.primary_vx)+" --steps=3000"
        lines = [
            "#!/bin/bash", "set -e", "cd " + shlex.quote(remote), "tar -xzf code_model.tar.gz",
            "cp -a " + shlex.quote(args.resource_source + "/resources") + " .",
            "rm -f code_model.tar.gz",
            *verification,
            "source /home/robot/Anaconda/etc/profile.d/conda.sh", "conda activate F1",
            "export PYTHONPATH=" + shlex.quote(remote),
            "xvfb-run -a " + base_command + " --seed=5 --reset_mode=standing --output_dir=" + primary_output + primary_speed + " > replay.log 2>&1"]
        for check in checks:
            speed_options = (" --vx=" + str(check["vx"]) + " --steps=" + str(check["steps"])) if "vx" in check else ""
            lines.append(base_command + " --no_video --seed=" + str(check["seed"]) +
                         " --reset_mode=" + check["reset_mode"] + " --output_dir=" + check["output"] +
                         speed_options +
                         " > " + Path(check["output"]).name + ".log 2>&1")
        lines.extend(["printf 'REPLAY_SUITE_OK\\n'", ""])
        script.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        run([SSH, "-o", "BatchMode=yes", HOST, "mkdir -p " + remote])
        for path in (archive, script):
            run([SCP, "-o", "BatchMode=yes", str(path), HOST + ":" + remote + "/" + path.name])
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        launch = ("nohup bash " + shlex.quote(remote+"/run_replay.sh") + " > "
                  + shlex.quote(remote+"/replay_suite.log") + " 2>&1 < /dev/null & echo $!")
        result = subprocess.run([SSH, "-o", "BatchMode=yes", HOST,
                                 "bash -c "+shlex.quote(launch)],
                                check=True, capture_output=True, text=True, timeout=30)
        pid = result.stdout.strip()
        if not pid.isdigit():
            raise RuntimeError("Inspect recorded workspace; replay launcher PID not observed")
        state.update(launch_pid=pid, phase="launched")
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
