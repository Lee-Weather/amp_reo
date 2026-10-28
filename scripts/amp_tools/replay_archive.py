"""Deploy the current AMP play code, keeping temporary files out of analysis."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resource_source", required=True)
    parser.add_argument("--task", choices=("x1_amp", "x1_amp_exp11", "x1_amp_exp3", "x1_amp_exp14", "x1_amp_exp15", "x1_amp_exp16", "x1_amp_exp17", "x1_amp_exp18", "x1_amp_exp19", "x1_amp_exp110", "x1_amp_exp111"), default="x1_amp")
    parser.add_argument("--training-task-id", default="TASK_20261022_177")
    parser.add_argument("--record_dir", type=Path)
    parser.add_argument("--extra_seeds", type=int, nargs="*", default=[])
    parser.add_argument("--check_rsi", action="store_true")
    parser.add_argument("--check_speeds", type=float, nargs="*", default=[],
                        help="Additional constant-vx checks with standing reset and seed=5")
    parser.add_argument("--speed_check_steps", type=int, default=1000,
                        help="Steps per constant-speed check (default: 20 seconds at 50 Hz)")
    args = parser.parse_args()
    if args.speed_check_steps < 1000:
        raise ValueError("Constant-speed checks require at least 1000 steps")
    if any(not math.isfinite(vx) or not -.2 <= vx <= .65 for vx in args.check_speeds):
        raise ValueError("Speed checks must stay within the trained vx range [-0.2, 0.65]")
    if not re.fullmatch(r"exp[1-9][0-9]*(?:\.[1-9][0-9]*)?|exp_[0-9]+(?:\.[0-9]+)?", args.experiment):
        raise ValueError("Use an experiment name such as exp1.4 (or a historical exp_N name)")
    model = args.model.resolve()
    import torch
    iteration = int(torch.load(model, map_location="cpu", weights_only=False)["iter"])
    if iteration < 0:
        raise ValueError("Checkpoint iteration must be nonnegative")
    model_name = "model_{}.pt".format(iteration)
    artifacts = args.record_dir.resolve() if args.record_dir else ROOT / "czy/exp1/artifacts" / args.experiment
    artifacts.mkdir(parents=True, exist_ok=True)
    state_file = artifacts / "replay_job.json"
    if state_file.exists():
        raise RuntimeError("An existing replay job is recorded; inspect its status before starting another")
    now = datetime.now(timezone(timedelta(hours=8)))
    remote = "/home/robot/czy/exp1/" + now.strftime("exp_%Y%m%d_%H%M%S")
    code = sorted((ROOT / "humanoid").rglob("*.py"))
    code_hash = hashlib.sha256()
    for path in code:
        code_hash.update(path.relative_to(ROOT).as_posix().encode())
        code_hash.update(path.read_bytes())
    primary_output = "logs/replay/standing_seed5"
    checks = [{"seed": seed, "reset_mode": "standing", "output": "logs/replay/standing_seed"+str(seed)}
              for seed in args.extra_seeds]
    if args.check_rsi:
        checks.append({"seed": 5, "reset_mode": "rsi", "output": "logs/replay/rsi_seed5"})
    for index, vx in enumerate(args.check_speeds):
        checks.append({"seed": 5, "reset_mode": "standing", "vx": vx,
                       "steps": args.speed_check_steps,
                       "output": "logs/replay/speed_check_{}".format(index)})
    state = {"experiment": args.experiment, "task_id": args.training_task_id, "task": args.task, "server_dir": remote,
             "checkpoint_iteration": iteration, "checkpoint_name": model_name,
             "created_at": now.isoformat(), "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
             "code_sha256": code_hash.hexdigest(), "profile": [0, .4, .6, 0, .4, 0],
             "stage_seconds": 10, "steps": 3000, "video": "1920x1080@25fps",
             "primary_output": primary_output, "checks": checks}
    def run(command):
        subprocess.run(command, check=True)
    with tempfile.TemporaryDirectory(prefix="x1_amp_replay_") as temp:
        archive = Path(temp) / "code_model.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in code:
                stream.add(path, arcname=path.relative_to(ROOT).as_posix())
            stream.add(ROOT / "setup.py", arcname="setup.py")
            stream.add(model, arcname="checkpoints/"+model_name)
        script = Path(temp) / "run_replay.sh"
        base_command = ("python -u humanoid/scripts/play_amp.py --task="+args.task+
                        " --headless --num_envs=1 --checkpoint_path=checkpoints/"+model_name)
        lines = [
            "#!/bin/bash", "set -e", "cd " + shlex.quote(remote), "tar -xzf code_model.tar.gz",
            "cp -a " + shlex.quote(args.resource_source + "/resources") + " .",
            "rm -f code_model.tar.gz",
            "source /home/robot/Anaconda/etc/profile.d/conda.sh", "conda activate F1",
            "export PYTHONPATH=" + shlex.quote(remote),
            "xvfb-run -a " + base_command + " --seed=5 --reset_mode=standing --output_dir=" + primary_output + " > replay.log 2>&1"]
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
        run([SSH, "-o", "BatchMode=yes", HOST,
             "nohup bash " + remote + "/run_replay.sh > " + remote + "/replay_suite.log 2>&1 < /dev/null &"])
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
