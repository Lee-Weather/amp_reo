"""Read compact task status and sanitized log tail for the current experiment."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone, timedelta

import flux_ops
from flux_ops import CLI, call, credentials, sanitize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Archived flux_training.json")
    parser.add_argument('--compact', action='store_true', help='Save one structured evaluation snapshot, without a full log file')
    args = parser.parse_args()
    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    flux_ops.ACCOUNT = state.get("account", flux_ops.ACCOUNT)
    task = state["task_id"]
    info = call("task", "info", "--task-id", task)["data"]
    base = info["taskBaseInfo"]
    summary = {key:base.get(key) for key in ("taskId", "taskName", "taskStatus", "startTime", "endTime", "runtime", "queue", "imageVersion", "goodsName")}
    summary["commit"] = info["taskCodeInfo"].get("commitId")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    env = os.environ.copy()
    env.update(FLUX_API_KEY=credentials()["api_key"],FLUX_BASE_URL=os.environ.get('FLUX_LOCAL_BASE_URL', "https://internal.limxdynamics.com/dev-api"))
    proc = subprocess.run([str(CLI), "--yes", "--timeout", "30s", "--retry", "1", "task", "logs",
                           "--task-id", task, "--raw", "--no-request-log"],
                          env=env,capture_output=True,encoding="utf-8",errors="replace")
    if args.compact:
        if proc.returncode: raise RuntimeError('Task log request failed')
        text = proc.stdout
        updates = list(re.finditer(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', text))
        latest = text[updates[-1].start():] if updates else text
        def metric(label):
            matches = re.findall(r'^\s*'+re.escape(label)+r':\s*([-+0-9.eE]+)', latest, re.M)
            return float(matches[-1]) if matches else None
        snapshot = dict(task_id=task, account=state['account'], commit=summary['commit'],
            checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), platform_status=str(base['taskStatus']),
            start_time=base.get('startTime'), iteration=int(updates[-1][1]) if updates else 0,
            target_iteration=state['target_iteration'], total_timesteps=metric('Total timesteps'),
            mean_reward=metric('Mean reward'), mean_episode_length_steps=metric('Mean episode length'),
            iteration_seconds=metric('Iteration time'), amp_curriculum_weight=metric('AMP/curriculum_weight'),
            forward_vx=metric('Tracking/forward_vx'), zero_applied_fraction=metric('Commands/zero_applied_fraction'),
            guide_weight=metric('Reference/guide_weight'),
            error_lines=[sanitize(line) for line in text.splitlines() if re.search(r'Traceback|\w+Error:', line)][-5:])
        assert snapshot['commit'] in ('', None, state['commit'])
        directory = state_path.parent/(state['experiment']+'_evaluation')
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'runtime_progress.json').write_text(json.dumps(snapshot, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(snapshot), flush=True)
        return
    if proc.returncode:
        print("LOGS_NOT_READY exit="+str(proc.returncode))
    else:
        text = proc.stdout.replace("[SDK]", "\n[SDK]").replace("Learning iteration ", "\nLearning iteration ")
        lines = text.splitlines()
        safe = []
        for line in lines:
            if re.search(r"(?i)password|secret|token|api.?key|authorization|startscript|asset_url_b64|aHR0cHM6Ly9",line):
                continue
            safe.append(sanitize(line))
        print("LOG_TAIL",len(lines),"total lines",flush=True)
        print("\n".join(safe[-45:])[-14000:],flush=True)
        # Preserve only sanitized log content in the already-ignored experiment directory.
        state_path.with_suffix(".log").write_text("\n".join(safe),encoding="utf-8")
    state.update(status=summary,latest_log_lines=safe[-10:] if not proc.returncode else [])
    state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    main()
