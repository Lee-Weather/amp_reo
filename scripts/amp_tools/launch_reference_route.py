"""Launch one explicitly authorized exp1.14 route on its published branch."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import flux_ops as flux

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "limx2f16enckjjsg@maxxspace.com"


def identifiers(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "taskId" and isinstance(item, str):
                yield item
            else:
                yield from identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from identifiers(item)
    elif isinstance(value, str) and value.startswith("TASK_"):
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("114", "114t"), required=True)
    args = parser.parse_args()
    experiment = "exp1.14"+("t" if args.route.endswith("t") else "")
    task = "x1_amp_exp"+args.route
    records = ROOT/"czy/exp1/records"
    state_file = records/(experiment+"_training.json")
    if state_file.exists():
        raise ValueError("Task state already exists; inspect the same task instead of creating another")
    evaluation = records/(experiment+"_evaluation")
    plan = json.loads((evaluation/"plan.json").read_text(encoding="utf-8"))
    publication = json.loads((evaluation/"publication.json").read_text(encoding="utf-8"))
    assert publication["published"] and publication["production_checkpoint_preflight"]
    assert plan["training_account"] == ACCOUNT and plan["additional_updates"] == 6000
    assert hashlib.sha256((ROOT/plan["source_model"]).read_bytes()).hexdigest() == plan["source_sha256"]
    git_env = os.environ.copy()
    git_env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    remote = subprocess.check_output(["git", "-c", "http.proxy="+publication["proxy"],
        "-c", "http.sslBackend=openssl", "ls-remote", "--heads", "origin", publication["branch"]], env=git_env).decode().split()[0]
    assert remote == publication["commit"]
    flux.ACCOUNT = ACCOUNT
    who = flux.call("auth", "whoami")["data"]
    assert who["email"] == ACCOUNT and int(who["userId"]) == 4681
    project = flux.call("project", "info", "--project-id", plan["project_id"])["data"]
    assert int(project["userId"]) == 4681
    existing = flux.call("task", "list", "--project-id", plan["project_id"], "--task-name", experiment+"-", "--limit", "50")["data"]["rows"]
    assert not existing, "Matching platform task already exists; inspect it before any creation"
    source_info = flux.call("task", "info", "--task-id", "TASK_20261029_318")["data"]
    assert int(source_info["taskBaseInfo"]["userId"]) == 4681
    rows = flux.call("task", "model", "list", "--task-id", "TASK_20261029_318", "--checkpoint", "6000", "--limit", "30")["data"]["rows"]
    source = next(r for r in rows if r["fileName"] == "model_6000.pt" and str(r["checkpoint"]) == "6000")
    assert source["policUrlDown"].startswith("https://")
    directory = "/workspace/amp_reo/logs/"+task+"/exported_data/"+experiment+"_main"
    command = ("gm-run amp_reo/humanoid/scripts/train.py --task="+task+
        " --headless --num_envs=4096 --seed=5 --max_iterations=6000 --run_name="+experiment+
        " --checkpoint_dir="+directory+" --warm_start="+directory+"/input_exp113t_model_6000.pt --warm_start_sha256="+plan["source_sha256"])
    encoded = base64.urlsafe_b64encode(source["policUrlDown"].encode()).decode()
    payload = {"taskBaseInfo": {"projectId": plan["project_id"], "taskType": "1", "trainType": "1",
        "taskName": experiment+"-"+("phase-reference" if args.route == "114" else "transition-contrast")+"-4096-seed5-plus6000",
        "taskDescription": "Transfer stable exp1.13t final6000 policy; fresh critic/D/Adam/replay; "+
            ("append phase to 98 features; dense q/dq/feet guidance" if args.route == "114" else "96 features unchanged; real-transition similarity minus held-leg baseline")+
            "; full standing costs; 24 rollout, 2x4 PPO, D4; 6000 additional updates.",
        "taskTag": [experiment, "AMP", "reference-guidance"], "goodsId": "ESKU000001", "imageId": "BJX00000001",
        "imageVersion": "V000412", "personalDataPath": ""},
        "taskCodeInfo": {"codeType": "2", "codeUrl": json.dumps([{"codeUrl": "https://github.com/Lee-Weather/amp_reo.git",
            "versionType": "1", "versionName": publication["branch"]}]), "mainCodeUri": "amp_reo/humanoid/scripts/train.py",
            "hparamsPath": "amp_reo/humanoid/envs/x1/"+task+"_config.py", "startScript": command+" --warm_start_url_b64="+encoded,
            "isOpen": "0"}, "runtimeReminderConfig": {"enableRuntimeReminder": False, "reminderDurations": []}}
    state = {"experiment": experiment, "account": ACCOUNT, "account_identity_verified": {"email": ACCOUNT, "user_id": 4681},
        "task": task, "branch": publication["branch"], "commit": publication["commit"], "project_id": plan["project_id"],
        "source_task_id": "TASK_20261029_318", "source_checkpoint": plan["source_model"], "source_iteration": 6000,
        "source_checkpoint_sha256": plan["source_sha256"], "training_mode": "policy_transfer_fresh_critic_amp_optimizers",
        "new_initial_iteration": 0, "max_iterations": 6000, "target_iteration": 6000, "num_envs": 4096, "seed": 5,
        "num_steps_per_env": 24, "num_learning_epochs": 2, "num_mini_batches": 4, "discriminator_updates_per_iteration": 4,
        "sample_budget": 589824000, "observation_dim": 6468 if args.route == "114" else 6336, "action_dim": 29,
        "checkpoint_dir": directory, "start_command": command, "first_checkpoint_iteration": 100,
        "goods_id": "ESKU000001", "image_id": "BJX00000001", "image_version": "V000412", "phase": "validated_not_created"}
    with tempfile.TemporaryDirectory(prefix=".flux-reference-", dir=ROOT) as temp:
        request = Path(temp)/"create.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        relative = request.relative_to(ROOT).as_posix()
        assert flux.call("--dry-run", "task", "create", "--file", relative)["success"]
        created = flux.call("task", "create", "--file", relative)
        request.unlink()
        ids = set(identifiers(created["data"]))
        if len(ids) != 1:
            raise RuntimeError("Creation succeeded but task ID needs inspection; do not create again")
        state.update(task_id=ids.pop(), phase="created", created_at=datetime.now(timezone.utc).isoformat())
        state_file.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        print("TASK_CREATED "+state["task_id"], flush=True)
    assert flux.call("--dry-run", "task", "run", "--task-id", state["task_id"])["success"]
    flux.call("task", "run", "--task-id", state["task_id"])
    state.update(phase="submitted", submitted_at=datetime.now(timezone.utc).isoformat())
    state_file.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
