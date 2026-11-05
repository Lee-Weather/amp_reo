"""Capture early runtime markers and verify a saved/uploaded output checkpoint."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

import flux_ops
from fetch_final_amp import save_receipt

ROOT = Path(__file__).resolve().parents[2]
MARKERS = ("CHECKPOINT_INPUT_READY", "CHECKPOINT_RESTORED", "CLOUD_RECOVERY_RESTORED",
           "CONTACT_RANDOMIZATION_READY",
           "NOMINAL_STARTUP_READY", "NOMINAL_STARTUP_RESET_VERIFIED", "NOMINAL_STARTUP_OBS_VERIFIED",
           "FRESH_START_OK", "POLICY_TRANSFER_OK", "TEACHER_TRANSFER_OK", "REWARD_REFINEMENT_OK",
           "ANCHORED_CONTINUATION_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--first-iteration", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    iteration = args.first_iteration or state.get("first_checkpoint_iteration", 100)
    filename = "model_{}.pt".format(iteration)
    flux_ops.ACCOUNT = state["account"]
    who = flux_ops.call("auth", "whoami")["data"]
    if who["email"] != state["account"]:
        raise ValueError("First-checkpoint account mismatch")
    env = os.environ.copy()
    env.update(FLUX_API_KEY=flux_ops.credentials()["api_key"], FLUX_BASE_URL=os.environ.get('FLUX_LOCAL_BASE_URL', "https://internal.limxdynamics.com/dev-api"))
    directory = ROOT/"czy/exp1/records"/(state["experiment"]+"_evaluation")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/"first_checkpoint_verification.json"
    evidence = {"task_id": state["task_id"], "account": state["account"], "first_iteration": iteration,
                "commit": state["commit"], "startup_markers": {}, "saved": False,
                "sdk_upload_confirmed": False, "platform_model_confirmed": False, "verified": False}
    deadline = time.monotonic()+args.timeout_seconds
    while time.monotonic() < deadline:
        try:
            info = flux_ops.call("task", "info", "--task-id", state["task_id"])["data"]
            if str(info["taskBaseInfo"]["userId"]) != str(who["userId"]):
                raise ValueError("Training task belongs to a different account")
            if info["taskCodeInfo"].get("commitId") not in (None, "", state["commit"]):
                raise ValueError("First-checkpoint training commit mismatch")
            if str(info["taskBaseInfo"]["taskStatus"]) == "6":
                raise ValueError("Task is terminal; inspect the same task, never restart automatically")
            log = subprocess.run([str(flux_ops.CLI), "--yes", "--timeout", "30s", "--retry", "1",
                                  "task", "logs", "--task-id", state["task_id"], "--raw", "--no-request-log"],
                                 env=env, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
            for marker in MARKERS:
                for match in re.finditer(marker+r"\s+", log):
                    try:
                        value, _ = json.JSONDecoder().raw_decode(log[match.end():])
                        evidence["startup_markers"][marker] = value
                    except ValueError:
                        pass
            updates = list(re.finditer(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", log))
            if updates:
                evidence["latest_iteration"] = int(updates[-1][1])
            try:
                evidence["save_receipt"] = save_receipt(log, iteration)
                evidence["saved"] = True
            except RuntimeError:
                pass
            if re.search(r"PT file \("+re.escape(filename)+r"\) uploaded successfully", log):
                evidence["sdk_upload_confirmed"] = True
            if evidence["saved"]:
                rows = flux_ops.call("task", "model", "list", "--task-id", state["task_id"],
                                     "--checkpoint", str(iteration), "--limit", "30")["data"]["rows"]
                exact = [r for r in rows if r.get("fileName") == filename and str(r.get("checkpoint")) == str(iteration)]
                if exact:
                    evidence["platform_model"] = {k: exact[0].get(k) for k in ("id", "fileName", "checkpoint", "createTime")}
                    evidence["platform_model_confirmed"] = True
            evidence.update(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                            actual_user_id=info["taskBaseInfo"]["userId"],
                            actual_commit=info["taskCodeInfo"].get("commitId"))
            evidence["verified"] = (evidence["saved"] and evidence["sdk_upload_confirmed"]
                                    and evidence["platform_model_confirmed"] and evidence["actual_commit"] == state["commit"])
            if state.get('teacher_sha256'):
                evidence['teacher_startup_verified'] = (evidence['startup_markers'].get('TEACHER_TRANSFER_OK', {}).get('source_sha256') == state['teacher_sha256'])
                evidence['verified'] = evidence['verified'] and evidence['teacher_startup_verified']
            if state.get('training_mode') == 'reward_refinement_preserve_policy_amp':
                marker = evidence['startup_markers'].get('REWARD_REFINEMENT_OK', {})
                evidence['refinement_startup_verified'] = (
                    marker.get('source_sha256') == state['source_checkpoint_sha256']
                    and marker.get('source_iteration') == marker.get('new_iteration') == state['source_iteration']
                    and marker.get('retained_disc_adam_step') == 4000
                    and 'disc_optimizer' in marker.get('retained', [])
                    and 'policy_optimizer' in marker.get('reset', []))
                evidence['verified'] = evidence['verified'] and evidence['refinement_startup_verified']
            if state.get('resume_mode') == 'full_checkpoint':
                marker=evidence['startup_markers'].get('CLOUD_RECOVERY_RESTORED',{})
                evidence['full_resume_startup_verified']=(
                    marker.get('source_sha256')==state['source_checkpoint_sha256']
                    and marker.get('source_iteration')==state['source_iteration']
                    and marker.get('target_iteration')==state['target_iteration']
                    and marker.get('additional_updates')==state['recovery_additional_updates']
                    and all(marker.get('optimizer_steps',{}).get(k,{}).get('minimum',0)>0 for k in ('policy','discriminator')))
                evidence['verified']=evidence['verified'] and evidence['full_resume_startup_verified']
                if state.get('refinement'):
                    evidence['refinement_startup_verified']=(marker.get('refinement')==state['refinement'])
                    evidence['verified']=evidence['verified'] and evidence['refinement_startup_verified']
                if state.get('refinement') in ('exp211_contact_only_v1','exp212_nominal_startup_v1','exp212_nominal_reset_fix_v1'):
                    evidence['contact_properties_verified']=evidence['startup_markers'].get('CONTACT_RANDOMIZATION_READY',{}).get('verified') is True
                    evidence['verified']=evidence['verified'] and evidence['contact_properties_verified']
                if state.get('refinement') in ('exp212_nominal_startup_v1','exp212_nominal_reset_fix_v1'):
                    markers=evidence['startup_markers']
                    ready=markers.get('NOMINAL_STARTUP_READY',{})
                    evidence['nominal_startup_verified']=(ready.get('verified') is True
                        and ready.get('nominal_envs')==state['num_envs']//4
                        and ready.get('fraction')==.25
                        and markers.get('NOMINAL_STARTUP_RESET_VERIFIED',{}).get('verified') is True
                        and markers.get('NOMINAL_STARTUP_OBS_VERIFIED',{}).get('verified') is True)
                    evidence['verified']=evidence['verified'] and evidence['nominal_startup_verified']
            elif state.get('training_mode') == 'anchored_continuation_preserve_critic_optimizers':
                marker=evidence['startup_markers'].get('ANCHORED_CONTINUATION_OK',{})
                evidence['continuation_startup_verified']=(
                    marker.get('source_sha256')==state['source_checkpoint_sha256']
                    and marker.get('source_iteration')==state['source_iteration']
                    and marker.get('new_iteration')==state.get('new_initial_iteration',0)
                    and all(k in marker.get('retained',[]) for k in ('critic','policy_optimizer','original7000_anchor'))
                    and marker.get('source_anchor_coef')==1.)
                evidence['verified']=evidence['verified'] and evidence['continuation_startup_verified']
            output.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
            if evidence["verified"]:
                current = json.loads(state_path.read_text(encoding="utf-8"))
                current["first_checkpoint_upload_verification"] = evidence
                state_path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                print("FIRST_CHECKPOINT_VERIFIED "+json.dumps(evidence), flush=True)
                return
        except ValueError:
            raise
        except Exception as exc:
            print("OBSERVATION_RETRY "+flux_ops.sanitize(str(exc))[-600:], flush=True)
        time.sleep(min(15, max(0, deadline-time.monotonic())))
    print("FIRST_CHECKPOINT_WAIT_EXPIRED_INSPECT_EXISTING_TASK", flush=True)


if __name__ == "__main__":
    main()
