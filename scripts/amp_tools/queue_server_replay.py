"""Queue final-model playback after a recorded SSH AMP run completes successfully."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from launch_server_amp import HOST, SCP, remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.state.read_text(encoding="utf-8"))
    if training["backend"] != "ssh_server" or training["host"] != HOST or training.get("preflight"):
        raise ValueError("Use an actual formal 201.5 training state")
    record = args.record_dir/"replay_job.json"
    if record.exists():
        raise FileExistsError("Inspect the existing queued playback; never create a duplicate")
    root = training["server_dir"]
    if not root.startswith("/home/robot/czy/exp1/exp_"):
        raise ValueError("Unexpected recorded training workspace")
    pid = int(training["launch_pid"])
    iteration = int(training["target_iteration"])
    now = datetime.now().astimezone()
    server = "/home/robot/czy/exp1/exp_"+now.strftime("%Y%m%d_%H%M%S")
    name = "model_{}.pt".format(iteration)
    checks = [dict(seed=5, reset_mode="rsi", output="logs/replay/rsi_seed5")]
    for index, vx in enumerate([-.2, .1, .65]):
        checks.append(dict(seed=5, reset_mode="standing", vx=vx, steps=1000,
                           output="logs/replay/speed_check_"+str(index)))
    for index, vx in enumerate([-.2, .1]):
        checks.append(dict(seed=5, reset_mode="rsi", vx=vx, steps=1000,
                           output="logs/replay/rsi_speed_check_"+str(index)))
    state = dict(experiment=training["experiment"], task_id="server_"+Path(root).name,
                 training_backend="ssh_server", training_state=args.state.as_posix(),
                 task=training["task"], server_dir=server, checkpoint_iteration=iteration,
                 checkpoint_name=name, checkpoint_sha256=None, code_sha256=training["code_sha256"],
                 code_hash_definition="SHA256 of sorted humanoid/*.py plus setup.py, with NUL separators",
                 profile=[0, .4, .6, 0, .4, 0], stage_seconds=10, steps=3000,
                 video="1920x1080@25fps", primary_output="logs/replay/standing_seed5", checks=checks,
                 phase="prepared", dependency_pid=str(pid), dependency_dir=root,
                 created_at=now.isoformat(), launch_pid=None)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    # Copy the exact training code, not a later local working-tree revision.
    preparation = "\n".join([
        "set -eu", "mkdir "+shlex.quote(server),
        "tar -C "+shlex.quote(root)+" --exclude=__pycache__ --exclude='*.pyc' -cf - humanoid setup.py"
        +" | tar -C "+shlex.quote(server)+" -xf -",
        "ln -s "+shlex.quote(training["resource_source"])+" "+shlex.quote(server+"/resources"), ""])
    remote("bash -s", preparation)
    subprocess.run([SCP, "-o", "BatchMode=yes", str(Path(__file__).with_name("replay_dependency.py")),
                    HOST+":"+server+"/replay_dependency.py"], check=True, timeout=45)
    dependency = ["python3", "replay_dependency.py", "--training-root="+root,
                  "--pid="+str(pid), "--iteration="+str(iteration), "--destination=checkpoints/"+name,
                  "--log-file="+training.get("log_file", "train_recovery.log"),
                  "--success-marker="+training.get("success_marker", "SERVER_RECOVERY_OK"),
                  "--launcher-script=run_train.sh", "--wait-gpu"]
    base = ["python", "-u", "humanoid/scripts/play_amp.py", "--task="+training["task"],
            "--headless", "--num_envs=1", "--seed=5", "--checkpoint_path=checkpoints/"+name]
    lines = ["set -euo pipefail", "cd "+shlex.quote(server), shlex.join(dependency),
             "source /home/robot/Anaconda/etc/profile.d/conda.sh", "conda activate F1",
             "export PYTHONPATH="+shlex.quote(server), "export OMP_NUM_THREADS=1",
             "xvfb-run -a "+shlex.join(base+["--reset_mode=standing", "--output_dir="+state["primary_output"]])
             +" > replay.log 2>&1"]
    for check in checks:
        command = base+["--no_video", "--reset_mode="+check["reset_mode"], "--output_dir="+check["output"]]
        if "vx" in check:
            command += ["--vx="+str(check["vx"]), "--steps="+str(check["steps"])]
        lines.append(shlex.join(command)+" > "+Path(check["output"]).name+".log 2>&1")
    lines += ["echo REPLAY_SUITE_OK", ""]
    remote("cat > "+shlex.quote(server+"/run_replay.sh"), "\n".join(lines))
    launch = "nohup bash "+shlex.quote(server+"/run_replay.sh")+" > "+shlex.quote(server+"/replay_suite.log")+" 2>&1 < /dev/null & echo $!"
    launcher = remote("bash -c "+shlex.quote(launch)).strip()
    if not launcher.isdigit():
        raise RuntimeError("Inspect the recorded workspace; launcher PID observation failed")
    state.update(launch_pid=launcher, phase="waiting_for_training")
    record.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
