"""Deploy the current AMP play code, keeping temporary files out of analysis."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
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
    args = parser.parse_args()
    if not args.experiment.startswith("exp_") or any(c not in "exp_0123456789." for c in args.experiment):
        raise ValueError("Use an exp_N archive name")
    model = args.model.resolve()
    artifacts = ROOT / "czy/exp1/artifacts" / args.experiment
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
    state = {"experiment": args.experiment, "task_id": "TASK_20261022_177", "server_dir": remote,
             "created_at": now.isoformat(), "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
             "code_sha256": code_hash.hexdigest(), "profile": [0, .4, .6, 0, .4, 0],
             "stage_seconds": 10, "steps": 3000, "video": "1920x1080@25fps"}
    def run(command):
        subprocess.run(command, check=True)
    with tempfile.TemporaryDirectory(prefix="x1_amp_replay_") as temp:
        archive = Path(temp) / "code_model.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in code:
                stream.add(path, arcname=path.relative_to(ROOT).as_posix())
            stream.add(ROOT / "setup.py", arcname="setup.py")
            stream.add(model, arcname="logs/x1_amp/exported_data/gm_play/model_6000.pt")
        script = Path(temp) / "run_replay.sh"
        script.write_text("\n".join([
            "#!/bin/bash", "set -e", "cd " + remote, "tar -xzf code_model.tar.gz",
            "cp -a " + args.resource_source + "/resources .",
            "source /home/robot/Anaconda/etc/profile.d/conda.sh", "conda activate F1",
            "export PYTHONPATH=" + remote,
            "exec xvfb-run -a python -u humanoid/scripts/play_amp.py --task=x1_amp --headless --num_envs=1 --seed=5 --checkpoint_path=logs/x1_amp/exported_data/gm_play/model_6000.pt --output_dir=logs/x1_amp/play_output/latest_play",
            ""]), encoding="utf-8", newline="\n")
        run([SSH, "-o", "BatchMode=yes", HOST, "mkdir -p " + remote])
        for path in (archive, script):
            run([SCP, "-o", "BatchMode=yes", str(path), HOST + ":" + remote + "/" + path.name])
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        run([SSH, "-o", "BatchMode=yes", HOST,
             "nohup bash " + remote + "/run_replay.sh > " + remote + "/replay.log 2>&1 < /dev/null &"])
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
