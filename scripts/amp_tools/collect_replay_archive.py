"""Collect and verify a finished six-stage AMP replay without temporary clutter."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from analyze_replay import load_context, evaluate_csv

ROOT = Path(__file__).resolve().parents[2]
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
HOST = "robot@10.12.201.5"


def collect_checks(state, remote, temp, context):
    reports = []
    for index, check in enumerate(state.get("checks", [])):
        case_dir = temp / ("check_"+str(index))
        case_dir.mkdir()
        for name in ("isaac_diag.csv", "replay_metadata.json"):
            subprocess.run([SCP, "-o", "BatchMode=yes", HOST + ":" + remote + "/" + check["output"] + "/" + name,
                            str(case_dir/name)], check=True)
        metadata = json.loads((case_dir/"replay_metadata.json").read_text())
        assert metadata["checkpoint"] == context["iteration"] and metadata["captured_steps"] == 3000
        assert metadata["seed"] == check["seed"] and metadata["reset_mode"] == check["reset_mode"]
        report = evaluate_csv(case_dir/"isaac_diag.csv", context)
        assert report["rows"] == 3000
        reports.append(dict(seed=check["seed"], reset_mode=check["reset_mode"], evaluation=report))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--record_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--primary_only", action="store_true", help="Archive the main trio while auxiliary checks run")
    mode.add_argument("--checks_only", action="store_true", help="Add finished auxiliary checks to an existing report")
    args = parser.parse_args()
    if not re.fullmatch(r"exp[1-9][0-9]*(?:\.[1-9][0-9]*)?|exp_[0-9]+(?:\.[0-9]+)?", args.experiment):
        raise ValueError("Use an experiment name such as exp1.4 (or a historical exp_N name)")
    artifacts = args.record_dir.resolve() if args.record_dir else ROOT / "czy/exp1/artifacts" / args.experiment
    state = json.loads((artifacts / "replay_job.json").read_text())
    context = load_context(args.model)
    iteration = state.get("checkpoint_iteration", 6000)
    assert context["iteration"] == iteration
    model_name = "model_{}.pt".format(iteration)
    remote = state["server_dir"]
    result = subprocess.run([SSH, "-o", "BatchMode=yes", HOST, "tail -n 1 " + remote + "/replay.log"],
                            check=True, capture_output=True, text=True)
    if "Replay files closed: 1500 frames" not in result.stdout:
        raise RuntimeError("Replay has not finished successfully; inspect the existing job, do not restart")
    destination = ROOT / "czy/data" / args.experiment
    if not args.checks_only and destination.exists() and any(p.name != model_name for p in destination.iterdir()):
        raise FileExistsError("Preserve existing replay: " + str(destination))
    existing_model = destination / model_name
    if existing_model.exists() and hashlib.sha256(existing_model.read_bytes()).hexdigest() != state["checkpoint_sha256"]:
        raise FileExistsError("Existing archive checkpoint differs from the replay model")
    if state.get("checks") and not args.primary_only:
        completed = subprocess.run([SSH, "-o", "BatchMode=yes", HOST,
                                    "tail -n 1 " + remote + "/replay_suite.log"], check=True, capture_output=True, text=True)
        if "REPLAY_SUITE_OK" not in completed.stdout:
            raise RuntimeError("Auxiliary replay checks have not finished")
    if args.checks_only:
        summary_path = artifacts / "replay_summary.json"
        summary = json.loads(summary_path.read_text())
        assert hashlib.sha256(args.model.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
        with tempfile.TemporaryDirectory(prefix="x1_amp_checks_") as temp:
            summary["auxiliary_checks"] = collect_checks(state, remote, Path(temp), context)
        summary["checks_pending"] = []
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Auxiliary checks added: " + str(len(summary["auxiliary_checks"])))
        return
    with tempfile.TemporaryDirectory(prefix="x1_amp_collect_") as temp:
        temp = Path(temp)
        for name in ("isaac_diag.csv", "play_output.mp4", "replay_metadata.json"):
            output = state.get("primary_output", "logs/x1_amp/play_output/latest_play")
            subprocess.run([SCP, "-o", "BatchMode=yes", HOST + ":" + remote + "/" + output + "/" + name, str(temp / name)], check=True)
        subprocess.run([SCP, "-o", "BatchMode=yes", HOST + ":" + remote + "/replay.log", str(temp / "replay.log")], check=True)
        metadata = json.loads((temp / "replay_metadata.json").read_text())
        assert metadata["captured_steps"] == 3000 and metadata["checkpoint"] == iteration
        assert metadata["video_frames"] == 1500 and len(metadata["segments"]) == 6
        with (temp / "isaac_diag.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3000
        numeric = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        assert all(np.isfinite(values).all() for values in numeric.values())
        assert np.max(np.abs(numeric["command_applied"] - numeric["command_observed"])) < 1e-6
        assert set(numeric["active_stage"].astype(int)) == set(range(6))
        video = cv2.VideoCapture(str(temp / "play_output.mp4"))
        assert video.isOpened()
        assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1920, 1080)
        fps = video.get(cv2.CAP_PROP_FPS)
        assert abs(fps - 25) < .001
        frames = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            assert frame.std() > 5
            frames += 1
        video.release()
        assert frames == 1500
        model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        assert model_hash == state["checkpoint_sha256"]
        done = numeric["done"].astype(bool)
        timeout = numeric["timeout"].astype(bool)
        summary = dict(rows=len(rows), duration_s=float(numeric["time_s"][-1]), decoded_frames=frames,
                       video_fps=fps, video_resolution=[1920, 1080], resets=int(done.sum()),
                       timeouts=int((done & timeout).sum()), non_timeout_terminations=int((done & ~timeout).sum()),
                       command_observation_max_error=float(np.max(np.abs(numeric["command_applied"]-numeric["command_observed"]))),
                       requested_applied_different_steps=int(np.count_nonzero(np.abs(numeric["command_requested"]-numeric["command_applied"]) > 1e-6)),
                       mean_task_reward=float(numeric["task_reward"].mean()),
                       mean_style_reward=float(numeric["style_reward"].mean()),
                       torque_clip_count=int(numeric["clip_count"][-1]), checkpoint_sha256=model_hash,
                       segments=metadata["segments"])
        summary["evaluation"] = evaluate_csv(temp / "isaac_diag.csv", context)
        if "style_score" in numeric:
            summary["mean_style_score"] = float(numeric["style_score"].mean())
        summary["auxiliary_checks"] = [] if args.primary_only else collect_checks(state, remote, temp, context)
        summary["checks_pending"] = state.get("checks", []) if args.primary_only else []
        destination.mkdir(parents=True, exist_ok=True)
        if args.model.resolve() != (destination / model_name).resolve():
            shutil.copy2(args.model, destination / model_name)
        for name in ("isaac_diag.csv", "play_output.mp4"):
            shutil.copy2(temp / name, destination / name)
        for name in ("replay_metadata.json", "replay.log"):
            shutil.copy2(temp / name, artifacts / name)
        summary["file_bytes"] = {path.name: path.stat().st_size for path in destination.iterdir()}
        (artifacts / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert sorted(p.name for p in destination.iterdir()) == ["isaac_diag.csv", model_name, "play_output.mp4"]
    print(json.dumps({key: value for key, value in summary.items() if key not in ("evaluation", "auxiliary_checks")}, indent=2))
    print("Detailed evaluation: " + str(artifacts / "replay_summary.json"))


if __name__ == "__main__":
    main()
