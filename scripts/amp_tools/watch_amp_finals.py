"""Wait on specific cloud tasks and collect their verified final checkpoints."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

import flux_ops
from fetch_final_amp import final_model

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--download-via-server", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.timeout_seconds <= 0:
        raise ValueError("Use a bounded wait with at least ten seconds between observations")
    output = args.pipeline.resolve()
    if ROOT not in output.parents:
        raise ValueError("Pipeline evidence must stay in the workspace")
    pipeline = json.loads(output.read_text()) if output.exists() else {}
    source_states = [path.resolve() for path in args.states]
    if any(ROOT not in path.parents for path in source_states):
        raise ValueError("Training state files must stay in the workspace")
    observed, downloaded = {}, set()
    deadline = time.monotonic()+args.timeout_seconds
    while time.monotonic() < deadline:
        observations = {}
        for state_path in source_states:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            experiment = state["experiment"]
            if experiment in downloaded:
                observations[experiment] = dict(observed[experiment], final_checkpoint_verified=True)
                continue
            flux_ops.ACCOUNT = state["account"]
            try:
                task = flux_ops.call("task", "info", "--task-id", state["task_id"])["data"]
                base = task["taskBaseInfo"]
                status = str(base["taskStatus"])
                if task["taskCodeInfo"]["commitId"] != state["commit"]:
                    raise ValueError("Observed training revision does not match the recorded task")
                result = {k: base.get(k) for k in ("taskId", "taskStatus", "startTime", "endTime", "runtime")}
                result["observed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
                result["final_checkpoint_verified"] = False
                if status == "6":
                    pipeline.update(phase="source_training_terminal_requires_review", sources={**observations, experiment: result})
                    output.write_text(json.dumps(pipeline, indent=2)+"\n", encoding="utf-8")
                    print("SOURCE_TERMINATED_REQUIRES_REVIEW "+json.dumps(result), flush=True)
                    return
                if status == "5":
                    rows = flux_ops.call("task", "model", "list", "--task-id", state["task_id"], "--checkpoint",
                                         str(state["target_iteration"]), "--limit", "30")["data"]["rows"]
                    try:
                        model = final_model(rows, state["target_iteration"])
                    except RuntimeError:
                        model = None
                    if model is not None:
                        command = [sys.executable, str(Path(__file__).with_name("fetch_final_amp.py")), "--state", str(state_path)]
                        if args.download_via_server:
                            command.append("--via-server")
                        # Child output never includes a URL; retain only a bounded failure tail.
                        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
                        if child.returncode:
                            raise RuntimeError(flux_ops.sanitize(child.stdout[-800:]+child.stderr[-1200:]))
                        result["final_checkpoint_verified"] = True
                        result["model_id"] = model["id"]
                        downloaded.add(experiment)
                        print("FINAL_CHECKPOINT_READY "+experiment+" "+child.stdout.strip(), flush=True)
                if experiment not in observed or observed[experiment]["taskStatus"] != status:
                    print("OBSERVED_SOURCE "+experiment+" "+json.dumps(result), flush=True)
                observations[experiment] = result
                observed[experiment] = result
            except Exception as exc:
                # Transient observation failures do not imply a stopped job.
                observations[experiment] = {"taskId": state["task_id"], "observation_error": str(exc),
                                            "last_successful_observation": observed.get(experiment)}
                print("OBSERVATION_RETRY "+experiment+" "+str(exc), flush=True)
        pipeline.update(phase="final_checkpoints_ready" if len(downloaded) == len(source_states) else "waiting_for_source_training",
                        sources=observations, checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
        output.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if len(downloaded) == len(source_states):
            print("BOTH_FINAL_CHECKPOINTS_READY", flush=True)
            return
        until = min(deadline, time.monotonic()+args.poll_seconds)
        while time.monotonic() < until:
            time.sleep(min(30, until-time.monotonic()))
    print("BOUNDED_WAIT_EXPIRED_CONTINUE_SAME_TASKS", flush=True)


if __name__ == "__main__":
    main()
