"""Launch an authorized seed5 AMP warm-start with one input/output directory."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import torch

import flux_ops

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"exp1\.[1-9][0-9]*", args.experiment):
        raise ValueError("Current experiments stay within exp1.x until user approves exp2")
    if not re.fullmatch(r"exp1\.[1-9][0-9]*_[A-Za-z0-9_]+", args.run_tag):
        raise ValueError("Use a unique tag such as exp1.4_upload_check")
    config_path = ROOT/"humanoid/envs/x1"/(args.task+"_config.py")
    if not re.fullmatch(r"x1_amp_exp[0-9]+", args.task) or not config_path.is_file():
        raise ValueError("Unknown local AMP task configuration")
    state_path = args.state.resolve()
    if ROOT not in state_path.parents or state_path.exists() or args.iterations <= 0:
        raise ValueError("Use a new state file inside this workspace and a positive update count")
    flux_ops.ACCOUNT = args.account
    call = flux_ops.call
    for probe in [("--help",), ("task", "--help"), ("project", "--help")]:
        subprocess.run([str(flux_ops.CLI), *probe], check=True, capture_output=True)
    identity = call("auth", "whoami")["data"]
    if identity["email"] != args.account:
        raise ValueError("Training account mismatch")
    source = call("task", "info", "--task-id", args.source_task)["data"]
    base = source["taskBaseInfo"]
    resources = call("task", "resource", "list", "--goods-back-category", "3", "--page", "1", "--limit", "20")["data"]["rows"]
    official = call("task", "image", "official")["data"]["rows"]
    versions = call("task", "image", "versions", "--image-id", base["imageId"])["data"]
    assert any(r["goodsId"] == base["goodsId"] for r in resources)
    assert any(r["imageId"] == base["imageId"] for r in official)
    assert any(r["id"] == base["imageVersion"] for r in versions)
    local_model = args.model.resolve()
    checkpoint = torch.load(local_model, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["iter"])
    digest = hashlib.sha256(local_model.read_bytes()).hexdigest()
    assert checkpoint["metadata"]["schema_version"] == 2
    assert len(checkpoint["metadata"]["dof_names"]) == 29
    model_name = "model_{}.pt".format(iteration)
    rows = call("task", "model", "list", "--task-id", args.source_task,
                "--checkpoint", str(iteration), "--limit", "30")["data"]["rows"]
    model = next(r for r in rows if r["fileName"] == model_name and str(r["checkpoint"]) == str(iteration))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "-c", "http.proxy=socks5h://10.12.201.122:39009",
                                      "ls-remote", "--heads", "origin", branch], cwd=ROOT, text=True)
    if remote.split()[0] != commit:
        raise ValueError("Push the intended training commit before launch")
    directory = "amp_reo/logs/{}/exported_data/{}".format(args.task, args.run_tag)
    absolute_directory = "/workspace/"+directory
    command = ("gm-run amp_reo/humanoid/scripts/train.py --task="+args.task+
               " --headless --num_envs=4096 --seed=5 --max_iterations="+str(args.iterations)+
               " --run_name="+args.experiment+" --checkpoint_dir="+absolute_directory+
               " --warm_start="+absolute_directory+"/"+model_name+" --warm_start_sha256="+digest)
    encoded = base64.urlsafe_b64encode(model["policUrlDown"].encode()).decode()
    payload = {
        "taskBaseInfo": {"projectId": base["projectId"], "taskType": "1", "trainType": "2",
                         "taskName": args.run_tag+"-4096-seed5-plus"+str(args.iterations),
                         "taskDescription": "AMP warm-start; shared input/output checkpoint directory; verified save receipts.",
                         "taskTag": [args.experiment, "AMP"], "goodsId": base["goodsId"],
                         "imageId": base["imageId"], "imageVersion": base["imageVersion"],
                         "personalDataPath": base.get("personalDataPath", "")},
        "taskCodeInfo": {"codeType": "2", "codeUrl": json.dumps([{"codeUrl": "https://github.com/Lee-Weather/amp_reo.git",
                                                                 "versionType": "1", "versionName": branch}]),
                         "mainCodeUri": "amp_reo/humanoid/scripts/train.py",
                         "hparamsPath": "amp_reo/"+config_path.relative_to(ROOT).as_posix(),
                         "startScript": command+" --warm_start_url_b64="+encoded,
                         "isOpen": "1", "checkPointFilePath": model["policUrl"],
                         "checkPointMountPath": directory+"/", "resumeFromTaskId": args.source_task,
                         "resumeFromTaskName": base["taskName"], "resumeFromCheckPoint": str(iteration)},
        "runtimeReminderConfig": {"enableRuntimeReminder": False, "reminderDurations": []}}
    state = {"experiment": args.experiment, "run_tag": args.run_tag, "account": args.account,
             "task": args.task, "branch": branch, "commit": commit, "project_id": base["projectId"],
             "source_task_id": args.source_task, "source_iteration": iteration,
             "source_checkpoint_sha256": digest, "target_iteration": iteration+args.iterations,
             "max_iterations": args.iterations, "num_envs": 4096, "seed": 5,
             "checkpoint_dir": absolute_directory, "start_command": command,
             "checkpoint_download_fallback": True, "phase": "validated_not_created"}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".flux_launch_", dir=ROOT) as temporary:
        request = Path(temporary)/"create.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        relative = request.relative_to(ROOT).as_posix()
        preview = call("--dry-run", "task", "create", "--file", relative)
        assert preview["success"]
        created = call("task", "create", "--file", relative)
        def task_ids(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "taskId" and isinstance(v, str):
                        yield v
                    else:
                        yield from task_ids(v)
            elif isinstance(value, list):
                for v in value:
                    yield from task_ids(v)
            elif isinstance(value, str) and re.fullmatch(r"TASK_[0-9]+_[0-9]+", value):
                yield value
        ids = set(task_ids(created["data"]))-{args.source_task}
        if len(ids) != 1:
            raise RuntimeError("Task created but ID response needs inspection; do not create again")
        state.update(task_id=ids.pop(), phase="created", created_at=datetime.now().astimezone().isoformat())
        state_path.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        print("TASK_CREATED "+state["task_id"], flush=True)
        call("task", "run", "--task-id", state["task_id"])
        state.update(phase="submitted", submitted_at=datetime.now().astimezone().isoformat())
        state_path.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
