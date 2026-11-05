"""Archive precise AMP learning curves and comparable completed-update windows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import flux_ops

ROOT = Path(__file__).resolve().parents[2]
KEYS = [
    "Train/mean_reward", "Train/mean_episode_length", "Policy/mean_noise_std",
    "AMP/style_score", "AMP/expert_score", "AMP/policy_score", "AMP/disc_loss",
    "AMP/grad_penalty", "AMP/task_reward", "AMP/style_reward", "AMP/kl_max",
    "AMP/state_estimator_loss", "AMP/policy_updates", "Loss/learning_rate",
    "Control/raw_target_clip_rate", "Control/torque_saturation_rate",
    "Tracking/forward_vx", "Tracking/forward_abs_vx_error",
    "Tracking/slow_forward_vx", "Tracking/slow_forward_abs_vx_error",
    "Tracking/backward_vx", "Tracking/backward_abs_vx_error",
    "Tracking/zero_abs_vx_error", "Tracking/lateral_abs_velocity",
    "Tracking/moving_abs_yaw_rate", "Init/rsi_early_failure_rate",
    "Init/standing_early_failure_rate", "Init/rsi_non_timeout_rate",
    "Init/standing_non_timeout_rate", "Curriculum/progress",
    "AMP/moving_style_multiplier", "AMP/standing_style_multiplier",
]


def summarize(series, low, high):
    result = {}
    for key, curve in series.items():
        points = [(step, value) for step, value in zip(curve["steps"], curve["values"])
                  if low <= step <= high]
        if points:
            result[key] = {"n": len(points), "mean": sum(v for _, v in points)/len(points),
                           "first_step": points[0][0], "last_step": points[-1][0]}
    return result


def merge_recovered_series(original, continued, source_iteration):
    """Discard unsaved source updates; retain only the restored weight path."""
    result = {}
    for key in sorted(set(original) | set(continued)):
        old, new = original.get(key, {}), continued.get(key, {})
        retained = [(s, v) for s, v in zip(old.get("steps", []), old.get("values", [])) if s <= source_iteration]
        resumed = list(zip(new.get("steps", []), new.get("values", [])))
        if any(s <= source_iteration for s, _ in resumed):
            raise ValueError("Recovery curves overlap or restart before the restored checkpoint")
        points = retained+resumed
        if points != sorted(points) or len({s for s, _ in points}) != len(points):
            raise ValueError("Recovery curve steps are not unique and ordered")
        result[key] = {"steps": [s for s, _ in points], "values": [v for _, v in points]}
    return result


def collect(state_path, max_workers=4):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    flux_ops.ACCOUNT = state["account"]
    identity = flux_ops.call("auth", "whoami")["data"]
    if identity["email"] != state["account"]:
        raise ValueError("Training curve account mismatch")
    info = flux_ops.call("task", "info", "--task-id", state["task_id"])["data"]
    if info["taskCodeInfo"]["commitId"] != state["commit"]:
        raise ValueError("Training curve source commit mismatch")
    now = datetime.now(timezone(timedelta(hours=8)))
    end_time = now.strftime("%Y-%m-%d %H:%M:%S")
    actual = flux_ops.call("task", "data", "keys", "--task-id", state["task_id"])["data"]
    keys = [key for key in KEYS if key in actual]

    def fetch(key):
        data = flux_ops.call("task", "data", "get", "--task-id", state["task_id"],
                             "--data-key", key, "--sampling-mode", "precise", "--end-time", end_time)["data"][key]
        steps, values = data["steps"], data["values"]
        if len(steps) != len(values) or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid training series: "+key)
        if len(set(steps)) != len(steps) or steps != sorted(steps):
            raise ValueError("Unordered or duplicate training steps: "+key)
        return key, {"steps": steps, "values": values}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        series = dict(pool.map(fetch, keys))
    segments = [{"task_id": state["task_id"], "commit": state["commit"]}]
    if state.get("resume_mode") == "full_checkpoint" and state.get("original_task_id"):
        original_path = ROOT/"czy/exp1/records"/(state["experiment"]+"_evaluation")/"recovery/original_training_curves.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if original["task_id"] != state["original_task_id"]:
            raise ValueError("Original recovery curve provenance mismatch")
        series = merge_recovered_series(original["series"], series, state["source_iteration"])
        segments[0]["retained_updates_after"] = state["source_iteration"]
        segments.insert(0, {"task_id": state["original_task_id"], "commit": state["original_commit"],
                            "retained_updates_through": state["source_iteration"]})
    if any(not series.get(key, {}).get("steps") for key in ("Train/mean_reward", "Train/mean_episode_length")):
        raise ValueError("Missing completed-episode statistics")
    # Every 'last100' metric covers the same update range, not different tails.
    # RSI or other conditional metrics can stop emitting after the curriculum
    # disables that branch. They must not move the final-update window back.
    common_end = min(series[key]["steps"][-1] for key in
                     ("Train/mean_reward", "Train/mean_episode_length"))
    windows = {"first200": (1, 200), "201-1500": (201, 1500),
               "1501-3000": (1501, 3000), "3001-4500": (3001, 4500),
               "4501-6000": (4501, 6000), "last100": (max(1, common_end-99), common_end)}
    origin = state.get("new_initial_iteration", 0)
    windows["session_first200"] = (origin+1, min(origin+200, common_end))
    summary = {"experiment": state["experiment"], "task_id": state["task_id"],
               "checked_at": now.isoformat(), "task_status": info["taskBaseInfo"]["taskStatus"],
               "commit": state["commit"], "sampling": "precise", "common_end": common_end,
               "source_segments": segments,
               "common_end_definition": "Last common reward/episode update; conditional metric endpoints do not truncate it",
               "latest_step_by_metric": {k:v["steps"][-1] for k,v in series.items() if v["steps"]},
               "missing_keys": sorted(set(KEYS)-set(keys)),
               "windows": {name: summarize(series, *bounds) for name, bounds in windows.items()},
               "limits": "Original runner completed-episode buffers overlap across updates. Episode duration includes reset standing buffer; it is not continuous walking duration."}
    output = ROOT/"czy/exp1/records"/(state["experiment"]+"_evaluation")
    output.mkdir(parents=True, exist_ok=True)
    (output/"training_curves.json").write_text(json.dumps({"checked_at": now.isoformat(),
        "task_id": state["task_id"], "sampling": "precise", "source_segments": segments,
        "series": series}, indent=2)+"\n", encoding="utf-8")
    (output/"training_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"experiment": state["experiment"], "common_end": common_end,
                      "series": len(series), "last100": summary["windows"]["last100"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for path in args.states:
        collect(path)
