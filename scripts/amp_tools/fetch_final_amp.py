"""Collect an uploaded final AMP checkpoint, including scratch and transfer runs."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import torch
import flux_ops

ROOT = Path(__file__).resolve().parents[2]
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
HOST = "robot@10.12.201.5"


def save_receipt(log, iteration):
    receipts = []
    for match in re.finditer(r"CHECKPOINT_SAVED\s+", log):
        try:
            value, _ = json.JSONDecoder().raw_decode(log[match.end():])
        except ValueError:
            continue
        if value.get("iteration") == iteration:
            if (not isinstance(value.get("bytes"), int) or value["bytes"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
                raise ValueError("Invalid final save receipt")
            receipts.append(value)
    if not receipts:
        raise RuntimeError("Final checkpoint save receipt is not yet visible")
    return receipts[-1]


def final_model(rows, iteration):
    matches = [row for row in rows if row.get("fileName") == "model_{}.pt".format(iteration)
               and str(row.get("checkpoint")) == str(iteration)]
    if not matches:
        raise RuntimeError("Exact final output is not uploaded yet; never substitute the input model")
    return max(matches, key=lambda row: int(row["id"]))


def stopped_model_iteration(rows, log, planned_iteration):
    """Select a real saved output after interruption, never the uploaded input."""
    updates = re.findall(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", log)
    if not updates:
        raise ValueError("Stopped task has no observable training updates")
    last = int(updates[-1][0])
    candidates = [int(row["checkpoint"]) for row in rows
                  if re.fullmatch(r"model_\d+\.pt", row.get("fileName", ""))
                  and row["fileName"] == "model_{}.pt".format(row.get("checkpoint"))
                  and 0 < int(row["checkpoint"]) <= min(last, planned_iteration)]
    if not candidates:
        raise ValueError("Stopped task has no uploaded training output checkpoint")
    return max(candidates), last


def validate_checkpoint(path, state):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["iter"] != state["target_iteration"]:
        raise ValueError("Checkpoint iteration is not the experiment's final iteration")
    meta, cfg = checkpoint["metadata"], checkpoint["config"]
    if (meta["num_obs"] != state["observation_dim"] or len(meta["dof_names"]) != state["action_dim"]
            or meta["policy"]["architecture"] != "ActorCriticDHAMP"
            or cfg["runner"]["experiment_name"] != state["task"]):
        raise ValueError("Checkpoint task/observation/policy identity does not match")
    origin = checkpoint.get("training_origin")
    if state["training_mode"] == "from_scratch":
        if origin is not None:
            raise ValueError("Scratch checkpoint unexpectedly has transferred weights")
    elif (not origin or origin.get("mode") != state["training_mode"]
          or origin.get("source_sha256") != state.get("policy_origin_source_sha256", state["source_checkpoint_sha256"])):
        raise ValueError("Transferred checkpoint lineage does not match")
    for group in ("model_state_dict", "discriminator_state_dict"):
        if not all(torch.isfinite(value).all() for value in checkpoint[group].values()):
            raise ValueError("Nonfinite checkpoint tensors")
    optimizer_steps = {}
    for name in ("optimizer_state_dict", "disc_optimizer_state_dict"):
        values = checkpoint[name]["state"]
        if not values:
            raise ValueError("Missing trained optimizer state")
        steps = [float(value["step"]) for value in values.values()]
        optimizer_steps[name] = {"parameters": len(values), "minimum_step": min(steps), "maximum_step": max(steps)}
    sys.path.insert(0, str(ROOT))
    from humanoid.amp.dh_models import ActorCriticDHAMP
    settings = dict(cfg["policy"])
    if settings.get("motion_exploration") is not None:
        exploration = dict(settings["motion_exploration"])
        exploration["indices"] = [meta["dof_names"].index(n) for n in exploration.pop("joint_names")]
        exploration["command_index"] = meta["num_obs"]-meta["policy"]["num_single_obs"]
        exploration["command_threshold"] = exploration.pop("speed_threshold")*cfg["normalization"]["obs_scales"]["lin_vel"]
        settings["motion_exploration"] = exploration
    env = cfg["env"]
    model = ActorCriticDHAMP(meta["num_obs"], env["num_privileged_obs"], state["action_dim"],
                            num_short_obs=env["short_frame_stack"]*env["num_single_obs"],
                            num_proprio_obs=env["num_single_obs"], **settings)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        output = model.act_inference(torch.zeros(2, meta["num_obs"]))
    if output.shape != (2, state["action_dim"]) or not torch.isfinite(output).all():
        raise ValueError("Final policy cannot produce finite actions")
    result = {"iteration": checkpoint["iter"], "metadata": meta, "training_origin": origin,
              "optimizer_steps": optimizer_steps, "inference_shape": list(output.shape),
              "config": cfg, "all_parameters_finite": True}
    if state.get("resume_mode") == "full_checkpoint":
        source_path = ROOT/state["source_checkpoint"]
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != state["source_checkpoint_sha256"]:
            raise ValueError("Recovery source checkpoint changed")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        expected_meta = source['metadata']
        if state.get('refinement') == 'exp210_yaw_only_v1':
            from humanoid.amp.exp210_transfer import expected_metadata, validate_configuration
            expected_meta = expected_metadata(expected_meta)
            validate_configuration(source['config'],cfg)
        if state.get('refinement') == 'exp211_contact_only_v1':
            from humanoid.amp.exp211_transfer import expected_metadata, validate_configuration
            expected_meta = expected_metadata(source['metadata'])
            validate_configuration(source['config'],cfg)
        if state.get('refinement') in ('exp212_nominal_startup_v1','exp212_nominal_reset_fix_v1'):
            from humanoid.amp.exp212_transfer import expected_metadata, validate_configuration
            expected_meta = expected_metadata(source['metadata'])
            validate_configuration(source['config'],cfg)
        updates = checkpoint["iter"]-source["iter"]
        if (not 0 < updates <= state["recovery_additional_updates"] or meta != expected_meta
                or origin != source.get("training_origin")):
            raise ValueError("Recovered model changed identity or did not advance the original budget")
        sections = ("policy", "algorithm", "discriminator", "env", "control", "sim", "amp", "commands", "noise", "domain_rand")
        checked_changes={'exp211_contact_only_v1':('domain_rand',),
                         'exp212_nominal_startup_v1':('domain_rand','amp')}
        protected_sections=tuple(k for k in sections if k not in checked_changes.get(state.get('refinement'),()))
        if any(cfg[key] != source["config"][key] for key in protected_sections):
            raise ValueError("Recovery changed training settings beyond its session/run budget")
        source_steps = {key: min(float(v["step"]) for v in source[key]["state"].values())
                        for key in optimizer_steps}
        maximum_ppo_updates = cfg["algorithm"]["num_learning_epochs"]*cfg["algorithm"]["num_mini_batches"]*updates
        old_policy=source['optimizer_state_dict']['state']
        new_policy=checkpoint['optimizer_state_dict']['state']
        if old_policy.keys()!=new_policy.keys():
            raise ValueError('Recovered policy Adam parameter mapping changed')
        policy_increments=[float(new_policy[k]['step'])-float(old_policy[k]['step']) for k in old_policy]
        if not (any(step>0 for step in policy_increments)
                and all(0<=step<=maximum_ppo_updates for step in policy_increments)):
            raise ValueError("Recovered policy Adam state did not continue")
        expected_disc_step = source_steps["disc_optimizer_state_dict"]+updates*cfg["discriminator"]["num_updates"]
        if (optimizer_steps["disc_optimizer_state_dict"]["minimum_step"] != expected_disc_step
                or optimizer_steps["disc_optimizer_state_dict"]["maximum_step"] != expected_disc_step):
            raise ValueError("Recovered discriminator Adam state did not continue")
        source_count = float(source["discriminator_state_dict"]["normalizer.count"])
        final_count = float(checkpoint["discriminator_state_dict"]["normalizer.count"])
        if final_count <= source_count:
            raise ValueError("Recovered discriminator normalizer did not continue")
        result["recovery_continuity"] = {"source_iteration": source["iter"], "additional_updates": updates,
            "source_optimizer_steps": source_steps, "expected_discriminator_step": expected_disc_step,
            "source_normalizer_count": source_count, "final_normalizer_count": final_count,
            "unchanged_config_sections": list(protected_sections),
            "limits": "Simulator/RNG/replay/episode buffers restart; no claim of uninterrupted trajectories."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--via-server", action="store_true", help="Download through the authorized 201.5 SSH host")
    parser.add_argument("--via-proxy", action="store_true", help="Download through the configured SOCKS proxy with TLS verification")
    parser.add_argument("--latest-stopped", action="store_true", help="Explicitly collect the latest real output from a stopped task; preserve its planned target")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    flux_ops.ACCOUNT = state["account"]
    who = flux_ops.call("auth", "whoami")["data"]
    if who["email"] != state["account"]:
        raise ValueError("Checkpoint source account mismatch")
    info = flux_ops.call("task", "info", "--task-id", state["task_id"])["data"]
    if str(info["taskBaseInfo"]["userId"]) != str(who["userId"]):
        raise ValueError("Final checkpoint task belongs to a different account")
    expected_status = "6" if args.latest_stopped else "5"
    if str(info["taskBaseInfo"]["taskStatus"]) != expected_status:
        raise RuntimeError("Training is not completed; keep waiting on the existing task")
    if info["taskCodeInfo"]["commitId"] != state["commit"]:
        raise ValueError("Completed task used a different training commit")
    env = os.environ.copy()
    env.update(FLUX_API_KEY=flux_ops.credentials()["api_key"], FLUX_BASE_URL=os.environ.get("FLUX_LOCAL_BASE_URL", "https://internal.limxdynamics.com/dev-api"))
    logs = subprocess.run([str(flux_ops.CLI), "--yes", "--timeout", "30s", "--retry", "1", "task", "logs",
                           "--task-id", state["task_id"], "--raw", "--no-request-log"],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    validation_state = dict(state)
    last_logged = None
    if args.latest_stopped:
        available = flux_ops.call("task", "model", "list", "--task-id", state["task_id"], "--limit", "100")["data"]["rows"]
        actual_iteration, last_logged = stopped_model_iteration(available, logs, state["target_iteration"])
        validation_state["target_iteration"] = actual_iteration
    else:
        actual_iteration = state["target_iteration"]
    receipt = save_receipt(logs, actual_iteration)
    rows = flux_ops.call("task", "model", "list", "--task-id", state["task_id"], "--checkpoint",
                         str(actual_iteration), "--limit", "30")["data"]["rows"]
    uploaded = final_model(rows, actual_iteration)
    url = uploaded["policUrlDown"]
    if not url.startswith("https://"):
        raise ValueError("Checkpoint download must use HTTPS")
    destination = ROOT/"czy/data"/state["experiment"]/uploaded["fileName"]
    if not re.fullmatch(r"exp[12]\.[0-9]+t?", state["experiment"]):
        raise ValueError("Unexpected experiment directory")
    with tempfile.TemporaryDirectory(prefix="amp_final_") as temporary:
        staged = Path(temporary)/uploaded["fileName"]
        if destination.exists():
            shutil.copy2(destination, staged)
        elif args.via_proxy:
            from private_training_input import request
            try:
                status, _, body = request(url)
                if status != 200:
                    raise ValueError("Unexpected checkpoint download status")
                staged.write_bytes(body)
            except Exception:
                raise RuntimeError("Checkpoint proxy download failed; signed URL withheld") from None
        elif args.via_server:
            # URL travels only on encrypted stdin, never shell argv or a local log.
            program = """import hashlib,json,pathlib,sys,tempfile,urllib.request
root=pathlib.Path(tempfile.mkdtemp(prefix='amp_final_',dir='/tmp')); path=root/'model.pt'
try:
 with urllib.request.urlopen(__URL__,timeout=120) as response,path.open('wb') as output:
  while True:
   block=response.read(1048576)
   if not block: break
   output.write(block)
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=__SHA__ or path.stat().st_size!=__BYTES__: raise ValueError('hash mismatch')
 print(json.dumps(dict(path=str(path),sha256=digest,bytes=path.stat().st_size)))
except Exception:
 path.unlink(missing_ok=True);root.rmdir();print('CHECKPOINT_DOWNLOAD_FAILED_URL_WITHHELD');sys.exit(1)
""".replace("__URL__", repr(url)).replace("__SHA__", repr(receipt["sha256"])).replace("__BYTES__", str(receipt["bytes"]))
            remote = subprocess.run([SSH, "-o", "BatchMode=yes", HOST, "python3 -"], input=program,
                                    text=True, capture_output=True, timeout=180, check=True)
            manifest = json.loads(remote.stdout)
            remote_path = manifest["path"]
            if not re.fullmatch(r"/tmp/amp_final_[A-Za-z0-9_]+/model\.pt", remote_path):
                raise ValueError("Unexpected temporary remote checkpoint path")
            try:
                subprocess.run([SCP, "-o", "BatchMode=yes", HOST+":"+remote_path, str(staged)], check=True, timeout=90)
            finally:
                subprocess.run([SSH, "-o", "BatchMode=yes", HOST,
                                "rm -f "+shlex.quote(remote_path)+" && rmdir "+shlex.quote(str(Path(remote_path).parent).replace('\\','/'))], check=True)
        else:
            import urllib.request
            try:
                with urllib.request.urlopen(url, timeout=90) as response, staged.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception:
                raise RuntimeError("Checkpoint download failed; signed URL withheld") from None
        if staged.stat().st_size != receipt["bytes"] or hashlib.sha256(staged.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Final model does not match its exact save receipt")
        torch.set_num_threads(1)
        validation = validate_checkpoint(staged, validation_state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(staged, destination)
    validation.update(save_receipt=receipt, model_id=uploaded["id"], bytes=receipt["bytes"], sha256=receipt["sha256"],
                      local_path=destination.relative_to(ROOT).as_posix(), training_commit=state["commit"],
                      verified_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    if args.latest_stopped:
        validation["interruption"] = {"planned_iteration": state["target_iteration"],
            "last_logged_iteration": last_logged, "latest_uploaded_iteration": actual_iteration,
            "unsaved_tail_updates": last_logged-actual_iteration,
            "reason": "User reports quota exhaustion; platform confirms stopped status6",
            "completed_planned_budget": False}
    record = ROOT/"czy/exp1/records"/(state["experiment"]+"_evaluation")/"checkpoint_verification.json"
    record.write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    state.update(phase="stopped_checkpoint_verified" if args.latest_stopped else "completed_checkpoint_verified", final_checkpoint=validation["local_path"],
                 final_checkpoint_sha256=receipt["sha256"], final_save_receipt=receipt,
                 status={k: info["taskBaseInfo"].get(k) for k in
                         ("taskId", "taskStatus", "startTime", "endTime", "runtime", "userId")})
    if args.latest_stopped:
        state.update(interruption=validation["interruption"], actual_checkpoint_iteration=actual_iteration,
                     platform_status="6", completed_planned_budget=False)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"experiment": state["experiment"], "iteration": validation["iteration"],
                      "path": validation["local_path"], "sha256": receipt["sha256"], "verified": True}))


if __name__ == "__main__":
    main()
