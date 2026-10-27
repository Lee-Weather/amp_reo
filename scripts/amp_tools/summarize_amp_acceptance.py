"""Apply recorded exp1.5/exp1.6 motion gates to a collected final replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.motion import URDFKinematics
from humanoid.amp.grounding import FootGrounding
from analyze_gait import geometry, metrics
from analyze_standing import analyze as analyze_standing
from analyze_replay import evaluate_csv, load_context


def braking_report(data, dt):
    """Require consecutive quiet control samples within one episode, after requested stops."""
    requested = data["command_requested"]
    starts = np.flatnonzero((abs(requested[1:]) <= 1e-6) & (abs(requested[:-1]) > 1e-6))+1
    hold = max(1, math.ceil(.2/dt-1e-6))
    reports = []
    for start in starts:
        stop_time = float(data["time_s"][start]-dt)
        following = np.flatnonzero(abs(requested[start:]) > 1e-6)
        end = start+following[0] if len(following) else len(data)
        streak, confirmation, entry, qualified = 0, None, None, 0
        for index in range(start, end):
            same_episode = index == start or data["episode_id"][index] == data["episode_id"][index-1]
            quiet = (not data["done"][index] and abs(data["command_applied"][index]) <= 1e-6
                     and np.hypot(data["vx"][index], data["vy"][index]) <= .02
                     and abs(data["wz"][index]) <= .03)
            if not same_episode:
                streak = 0
            streak = streak+1 if quiet else 0
            qualified += int(quiet)
            if streak >= hold and confirmation is None:
                confirmation = float(data["time_s"][index]-stop_time)
                entry = float(data["time_s"][index-hold+1]-stop_time)
        reports.append(dict(stage=int(data["active_stage"][start]), stop_command_time_s=stop_time,
                            entry_time_for_02s_hold_s=entry, confirmation_time_s=confirmation,
                            qualifying_fraction=qualified/(end-start), never_confirmed=confirmation is None))
    return dict(criterion="Planar norm <=0.02m/s, abs wz <=0.03rad/s for 0.2s of consecutive control samples in one episode",
                required_samples=hold, stops=reports)


def auxiliary_gates(checks, revision):
    required = [("rsi", None), ("standing", -.2), ("standing", .1), ("standing", .65)]
    if revision == "exp1.6":
        required += [("rsi", -.2), ("rsi", .1)]
    reports = []
    for mode, vx in required:
        matches = [c for c in checks if c["reset_mode"] == mode and c.get("vx") == vx and c["seed"] == 5]
        item = dict(reset_mode=mode, command_vx=vx, collected=len(matches) == 1, passed=False)
        if len(matches) == 1:
            c, e = matches[0], matches[0]["evaluation"]
            item["no_falls"] = e["non_timeout_terminations"] == 0
            enough = e["rows"] == (3000 if vx is None else 1000)
            item["passed"] = item["no_falls"] and enough
            if vx is not None:
                w = e["walking"]
                lo, hi, maximum_rmse = {-.2:(-.25,-.15,.08), .1:(.06,.14,.06), .65:(.55,.75,.12)}[vx]
                enough = w.get("samples", 0) >= 600
                item.update(mean_vx=w.get("mean_vx"), rmse=w.get("rmse_vx"), enough_settled_samples=enough)
                item["speed_mean_pass"] = enough and lo <= w["mean_vx"] <= hi
                item["rmse_pass"] = enough and w["rmse_vx"] <= maximum_rmse
                item["passed"] &= item["speed_mean_pass"] and item["rmse_pass"]
        reports.append(item)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--criteria-version", choices=("exp1.5", "exp1.6"), default="exp1.6")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual-review", type=Path, help="Review JSON bound to model/video hashes; absent means pending")
    args = parser.parse_args()
    torch.set_num_threads(1)
    context = load_context(args.model)
    meta = context["metadata"]
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    csv_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    video = args.csv.parent/"play_output.mp4"
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    summary = json.loads((args.record_dir/"replay_summary.json").read_text(encoding="utf-8"))
    replay_meta = json.loads((args.record_dir/"replay_metadata.json").read_text(encoding="utf-8"))
    if summary["checkpoint_sha256"] != digest or replay_meta["checkpoint"] != context["iteration"]:
        raise ValueError("Replay and model do not refer to the same checkpoint")
    if replay_meta["policy_metadata"] != meta or replay_meta["seed"] != 5 or replay_meta["noise"] or replay_meta["pushes"]:
        raise ValueError("Use the matching seed5, no-noise, no-push final replay")
    if replay_meta["reset_mode"] != "standing":
        raise ValueError("Primary replay must use standing resets")
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    dt = meta["policy_dt"]
    if len(data) != 3000 or not all(np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError("Require the full finite 3000-row replay")
    if not np.allclose(np.diff(data["time_s"]), dt, atol=1e-5, rtol=0):
        raise ValueError("Unexpected control sample spacing")
    evaluation = evaluate_csv(args.csv, context)
    try:
        windows = analyze_standing(args.csv)[-1]
    except ValueError:
        windows = {}
    standing = {k:dict(vx=v["mean_abs_vx_m_s"] <= .02, vy=v["mean_abs_vy_m_s"] <= .02,
                       wz=v["mean_abs_wz_rad_s"] <= .03, roll_span=v["roll_p95_p05_deg"] <= 2,
                       pitch_span=v["pitch_p95_p05_deg"] <= 2,
                       knees=all(v["joints"][s+"_knee_pitch_joint"]["q_p95_p05_deg"] <= 3 for s in ("left","right")))
                for k,v in windows.items()}
    mask = (data["time_s"] >= 23.6) & (data["time_s"] < 29.8)
    gait_valid = (not data["done"][mask].any() and len(np.unique(data["episode_id"][mask])) == 1
                  and np.allclose(data["command_applied"][mask], .6, atol=1e-6))
    names = meta["dof_names"]
    urdf = ROOT/"resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
    fk = URDFKinematics(urdf, names, [s+"_ankle_roll_link" for s in ("left","right")])
    root = torch.tensor(np.column_stack([data[k][mask] for k in ("x","y","z","qx","qy","qz","qw")]), dtype=torch.float32)
    q = torch.tensor(np.column_stack([data["q_"+n][mask] for n in names]), dtype=torch.float32)
    local, world, sole = geometry(root, q, fk, FootGrounding(urdf, fk, .002))
    logged = np.stack([np.column_stack([data["foot_"+a+"_"+s][mask] for a in "xyz"]) for s in ("l","r")], 1)
    stable = (data["episode_step"][mask] > 2) & ~data["done"][mask].astype(bool)
    fk_error = float(abs(world[stable]-logged[stable]).max()) if stable.any() else None
    if fk_error is not None and fk_error > 1e-4:
        raise ValueError("Foot FK does not match simulator diagnostics")
    gait = metrics(q.numpy(), local, sole, data["vx"][mask], names, dt)
    period = gait["same_leg_cycle_period_s"]
    count = round(period/dt) if period is not None and math.isfinite(period) and period > 0 else 0
    cycles = int(mask.sum())//count if count else 0
    bias = {k:float(abs(data[k][mask][:count*cycles].reshape(cycles,count).mean(1)).mean())
            for k in ("vy","wz")} if cycles >= 2 else {}
    braking = braking_report(data, dt)
    auxiliary = auxiliary_gates(summary.get("auxiliary_checks", []), args.criteria_version)
    criteria = dict(primary_no_falls=evaluation["non_timeout_terminations"] == 0,
                    strict_trio=sorted(p.name for p in args.csv.parent.iterdir()) == ["isaac_diag.csv",args.model.name,"play_output.mp4"],
                    artifact_hashes_bound=all(summary.get("file_sha256",{}).get(k) == v for k,v in
                                             ((args.model.name,digest),("isaac_diag.csv",csv_hash),("play_output.mp4",video_hash))),
                    gait_window_without_reset=gait_valid and fk_error is not None,
                    effective_targets_within_hard_limits=evaluation["criteria"]["effective_targets_within_hard_limits"] is True,
                    full_video_collected=summary["decoded_frames"] == 1500,
                    all_auxiliary_checks=all(c["passed"] for c in auxiliary) and not summary.get("checks_pending"),
                    speed06_rmse_le_010=evaluation["speed_06"].get("samples",0) > 0 and evaluation["speed_06"]["rmse_vx"] <= .10,
                    all_standing_windows=len(standing) == 3 and all(all(v.values()) for v in standing.values()),
                    both_stops_confirm_within_2s=len(braking["stops"]) == 2 and all(s["confirmation_time_s"] is not None and s["confirmation_time_s"] <= 2+1e-6 for s in braking["stops"]),
                    each_sole_p95_ge_15mm=all(f["sole_p95_m"] >= .015 for f in gait["feet"].values()),
                    each_foot_foreaft_ge_25cm=all(f["forward_excursion_p95_p05_m"] >= .25 for f in gait["feet"].values()),
                    cycle_averaged_abs_wz_le_005=bool(bias) and bias["wz"] <= .05)
    if args.criteria_version == "exp1.6":
        criteria.update(each_sole_below_1cm_fraction_le_085=all(f["sole_below_1cm_fraction"] <= .85 for f in gait["feet"].values()),
                        cycle_averaged_abs_vy_le_005=bool(bias) and bias["vy"] <= .05,
                        speed04_mean_error_le_008=evaluation["speed_04"].get("samples",0) > 0 and abs(evaluation["speed_04"]["mean_vx"]-.4) <= .08,
                        speed06_mean_error_le_008=evaluation["speed_06"].get("samples",0) > 0 and abs(evaluation["speed_06"]["mean_vx"]-.6) <= .08)
    review = json.loads(args.visual_review.read_text(encoding="utf-8")) if args.visual_review else {}
    visual_pass = (review.get("status") == "passed" and review.get("checkpoint_sha256") == digest
                   and review.get("video_sha256") == video_hash and bool(review.get("observations")))
    result = dict(criteria_version=args.criteria_version, checkpoint_iteration=context["iteration"], checkpoint_sha256=digest,
                  csv_sha256=csv_hash, video_sha256=video_hash,
                  numeric_criteria_passed=all(criteria.values()), visual_review_passed=visual_pass,
                  accepted=all(criteria.values()) and visual_pass, criteria=criteria,
                  standing_windows=standing, auxiliary_checks=auxiliary, braking=braking,
                  gait_summary=gait, foot_fk_max_error_m=fk_error,
                  cycle_diagnostic=dict(period_s=period, samples_per_cycle=count, cycles=cycles,
                                        omitted_samples=int(mask.sum())-count*cycles, mean_abs_cycle_average=bias,
                                        definition="Body-frame rates averaged over nonoverlapping complete measured-cycle bins"))
    output = args.output or args.record_dir/"acceptance_summary.json"
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(accepted=result["accepted"], criteria=criteria, output=str(output))))


if __name__ == "__main__":
    main()
