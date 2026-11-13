"""Summarise an Isaac replay diagnostic CSV, one block per ``active_stage``.

Usage:
    python scripts/amp_tools/analyze_isaac_diag.py --csv czy/data/exp3.1/isaac_diag.csv
    python scripts/amp_tools/analyze_isaac_diag.py --csv ... --json out.json --dt 0.02

The replay profile sets the command in stages; this tool reports, per stage, how
long the robot stayed up, how well it tracked, and how far it tilted, so the same
numbers can be quoted for every experiment.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

FIELDS = ("active_stage", "command_applied", "cmd_linear_x", "vx", "vy",
          "base_euler_x", "base_euler_y", "base_euler_z", "base_height",
          "done", "timeout", "episode_id")


def load(path):
    data = {name: [] for name in FIELDS}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in FIELDS if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing columns: {missing}")
        for row in reader:
            for name in FIELDS:
                raw = row[name]
                data[name].append(float(raw) if raw not in ("", "nan") else math.nan)
    return {name: np.asarray(values, dtype=float) for name, values in data.items()}


def stage_blocks(data, dt):
    stages = data["active_stage"].astype(int)
    command = data["cmd_linear_x"]
    applied = data["command_applied"]
    blocks, start = [], 0
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[start]:
            segment = slice(start, index)
            block = dict(stage=int(stages[start]),
                         steps=index - start,
                         seconds=round((index - start) * dt, 2),
                         cmd=round(float(np.nanmedian(command[segment])), 4),
                         applied_mean=round(float(np.nanmean(applied[segment])), 4),
                         resets=int(np.sum(data["done"][segment] > 0.5)),
                         timeouts=int(np.sum(data["timeout"][segment] > 0.5)))
            vx = data["vx"][segment]
            vy = data["vy"][segment]
            valid = np.isfinite(vx)
            block["vx_mean"] = round(float(np.mean(vx[valid])), 4) if valid.any() else None
            block["vx_rmse"] = round(float(np.sqrt(np.mean((vx[valid] - block["cmd"]) ** 2))), 4) if valid.any() else None
            block["abs_vy_mean"] = round(float(np.mean(np.abs(vy[valid]))), 4) if valid.any() else None
            yaw = data["base_euler_z"][segment]
            block["yaw_span_deg"] = round(float(np.nanmax(yaw) - np.nanmin(yaw)), 2)
            height = data["base_height"][segment]
            block["height_mean"] = round(float(np.nanmean(height)), 3)
            block["height_min"] = round(float(np.nanmin(height)), 3)
            roll = np.abs(data["base_euler_x"][segment])
            pitch = np.abs(data["base_euler_y"][segment])
            block["abs_roll_max_deg"] = round(float(np.nanmax(roll)), 2)
            block["abs_pitch_max_deg"] = round(float(np.nanmax(pitch)), 2)
            block["abs_pitch_mean_deg"] = round(float(np.nanmean(pitch)), 2)
            blocks.append(block)
            start = index
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    data = load(args.csv)
    blocks = stage_blocks(data, args.dt)
    report = dict(csv=str(args.csv), steps=int(len(data["vx"])),
                  resets=int(np.sum(data["done"] > 0.5)),
                  timeouts=int(np.sum(data["timeout"] > 0.5)),
                  episodes=int(len(np.unique(data["episode_id"]))),
                  stages=blocks)
    header = ("stage", "sec", "cmd", "applied", "resets", "timeouts",
              "vx_mean", "vx_rmse", "|vy|", "yaw°", "h_mean", "h_min",
              "|roll|max", "|pitch|max")
    print(" ".join(f"{name:>10}" for name in header))
    for block in blocks:
        values = (block["stage"], block["seconds"], block["cmd"],
                  block["applied_mean"], block["resets"], block["timeouts"],
                  block["vx_mean"], block["vx_rmse"], block["abs_vy_mean"],
                  block["yaw_span_deg"], block["height_mean"], block["height_min"],
                  block["abs_roll_max_deg"], block["abs_pitch_max_deg"])
        print(" ".join(f"{value:>10}" for value in values))
    print(f"total steps={report['steps']} resets={report['resets']} "
          f"timeouts={report['timeouts']} episodes={report['episodes']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()