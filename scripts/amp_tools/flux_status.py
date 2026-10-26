"""Read compact task status and sanitized log tail for the current experiment."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import flux_ops
from flux_ops import CLI, call, credentials, sanitize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Archived flux_training.json")
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
    env.update(FLUX_API_KEY=credentials()["api_key"],FLUX_BASE_URL="https://internal.limxdynamics.com/dev-api")
    proc = subprocess.run([str(CLI), "--yes", "--timeout", "30s", "--retry", "1", "task", "logs",
                           "--task-id", task, "--raw", "--no-request-log"],
                          env=env,capture_output=True,encoding="utf-8",errors="replace")
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
