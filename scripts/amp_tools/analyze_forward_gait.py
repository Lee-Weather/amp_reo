"""Diagnose a constant-speed primary replay against its speed-scaled reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_gait import geometry, metrics
from humanoid.amp.motion import MotionLibrary, URDFKinematics
from humanoid.amp.grounding import FootGrounding
from humanoid.amp.phase_reference import PhaseReference
from humanoid.amp.native_reference import NativeCycle


def describe(q, local, sole, vx, names, dt, force=None):
    result = metrics(q, local, sole, vx, names, dt, force)
    # Peak picking can skip alternating peaks. Averaging left/right peak
    # periods would invent a cycle that neither leg actually performed.
    result.pop("same_leg_cycle_period_s")
    result.pop("trunk_forward_distance_per_same_leg_cycle_m")
    selected = [s+"_"+j+"_joint" for s in ("left", "right")
                for j in ("hip_pitch", "knee_pitch")]
    hz = float(np.median([result["joints"][n]["frequency"]["dominant_hz"] for n in selected]))
    result["dominant_hip_knee_oscillation_hz"] = hz
    result["body_distance_per_dominant_joint_oscillation_m"] = float(vx.mean()/hz)
    for i, side in enumerate(("left", "right")):
        raised = sole[:, i] > .01
        edges = np.diff(np.r_[False, raised, False].astype(int))
        lengths = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
        result["feet"][side]["clearance_over_1cm_runs_at_least_80ms"] = int((lengths*dt >= .08-1e-7).sum())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vx", type=float)
    p.add_argument("--window-starts", type=float, nargs="+")
    p.add_argument("--window-seconds", type=float)
    a = p.parse_args()
    torch.set_num_threads(1)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    native = 'native_reference' in ck['metadata'].get('training',{})
    if a.vx is None: a.vx = ck['config']['commands']['native_speed'] if native else .3
    if a.window_starts is None: a.window_starts = [6.,26.,46.] if native else [4.,24.,44.]
    if a.window_seconds is None: a.window_seconds = 12. if native else 14.
    if not np.isfinite(a.vx) or a.vx <= 0 or a.window_seconds < 6:
        raise ValueError("Use a positive forward command and at least six seconds per continuous window")
    job_path = a.output_dir/"replay_job.json"
    source_verified = False
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if hashlib.sha256(a.model.read_bytes()).hexdigest() != job["checkpoint_sha256"]:
            raise ValueError("Analysis model differs from the recorded replay model")
        references = ['humanoid/amp/motion.py','humanoid/amp/grounding.py']
        references += ['humanoid/amp/native_reference.py','humanoid/amp/reference_data/exp20_native_correction.json',
            'humanoid/amp/reference_data/exp20_sole_hulls.json'] if native else ['humanoid/amp/phase_reference.py']
        for name in references:
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != job["source_file_sha256"][name]:
                raise ValueError("Analysis reference code differs from the replay's training revision")
        source_verified = True
    names, dt = ck["metadata"]["dof_names"], ck["metadata"]["policy_dt"]
    cfg = ck["config"]["amp"]
    with a.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if not all(np.isfinite(v).all() for v in d.values()):
        raise ValueError("Nonfinite primary CSV")
    if not np.allclose(d["command_requested"], a.vx, atol=1e-6, rtol=0):
        raise ValueError("This analysis requires a constant requested forward command")
    q = np.column_stack([d["q_"+n] for n in names])
    root = torch.tensor(np.column_stack([d[k] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]), dtype=torch.float32)
    urdf = ROOT/"resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
    fk = URDFKinematics(urdf, names, [s+"_ankle_roll_link" for s in ("left", "right")])
    ground = FootGrounding(urdf, fk, cfg["ground_clearance"])
    local, world, sole = geometry(root, torch.tensor(q, dtype=torch.float32), fk, ground)
    logged = np.stack([np.column_stack([d["foot_"+axis+"_"+s] for axis in "xyz"]) for s in ("l", "r")], 1)
    valid = (d["episode_step"] > 2) & ~d["done"].astype(bool)
    fk_error = float(np.abs(world[valid]-logged[valid]).max())
    if fk_error > 1e-4:
        raise ValueError("Reconstructed foot origins do not match simulator CSV")
    motion_dir = ROOT/"resources/motions/slow_walk_amp_v1.2/normalized"
    for stem, digest in ck["metadata"]["training"]["reference_source_sha256"].items():
        if hashlib.sha256((motion_dir/(stem+".pkl")).read_bytes()).hexdigest() != digest:
            raise ValueError("Reference resource differs from training checkpoint")
    lib = MotionLibrary(motion_dir, urdf, names, cfg["key_body_names"], dt=dt,
                        history_steps=cfg["history_steps"], weights=cfg["motion_weights"],
                        limit_tolerance=cfg["limit_tolerance"], ground_clearance=cfg["ground_clearance"])
    clip = lib.names.index(cfg["standing_clip"])
    standing = lib.state(torch.tensor([clip]), torch.tensor([cfg["standing_frame"]/float(lib.fps[clip])]))[0]
    if native:
        path = Path(cfg['native_correction_file'].format(LEGGED_GYM_ROOT_DIR=ROOT))
        correction = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(correction,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=ck['metadata']['training']['native_reference']['correction_sha256']:
            raise ValueError('Native reference correction differs from checkpoint')
        reference = NativeCycle(lib,cfg['native_clip'],cfg['native_first_frame'],cfg['native_last_frame'],contact_correction=correction)
    else:
        reference = PhaseReference(lib, standing[13:42], cfg["reference_cycles"])
    rate = float(reference.phase_rate(torch.tensor([a.vx]))[0])
    time = np.arange(0, a.window_seconds, dt)
    rq, _ = reference.sample(torch.full((len(time),), a.vx), torch.tensor(time*rate, dtype=torch.float32))
    if native:
        state = reference.state(torch.tensor(time*rate,dtype=torch.float32))
        rl,_,rs = geometry(state[:,:7],rq,fk,ground)
    else:
        rotation = standing[3:7].expand(len(time), -1)
        position = ground.align(torch.zeros(len(time), 3), rotation, rq)
        rl, _, rs = geometry(torch.cat((position, rotation), -1), rq, fk, ground)
    report = dict(checkpoint_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(), command_vx=a.vx,
                  reference_source_code_verified=source_verified,
                  foot_fk_max_error_m=fk_error, reference_phase_rate_hz=rate, reference_period_s=1/rate,
                  reference=describe(rq.numpy(), rl, rs, np.full(len(time), a.vx), names, dt),
                  policy_windows={}, definitions={
                      "windows": "Continuous, nonterminal, same-episode applied-command windows; excluded reset and standing buffer.",
                      "frequency": "Median dominant PSD frequency of the two hips and knees. Joint oscillation, not touchdown step count; harmonics may be present.",
                      "excursion": "P95-P05 of foot link origin relative to pelvis. Not touchdown step length.",
                      "sole_height": "Lowest collision-mesh vertex, including foot orientation; actual geometry checked against simulator foot origins.",
                      "reference": ('Checkpoint native full-state cycle with its named contact correction.' if native else "Actual checkpoint PhaseReference at requested speed, not the faster unscaled B3 clip."),
                      "reference_sole": ('Native full-state root pose and collision soles; kinematic, not physics.' if native else "Kinematic reference q with demonstrated standing root orientation, grounded to 2mm each frame; not a physics rollout."),
                      "phase_limit": ("CSV records instantaneous reference_phase/q/dq; reward formulas are task-specific."
                                      if "reference_phase" in d else
                                      "CSV does not record instantaneous gait phase; frequency mismatch is evidence, not a measured phase-error curve."),
                      "scope": "Only the existing primary forward replay. No auxiliary replay analyzed or new simulation launched."})
    force = np.column_stack([d["foot_force_"+s] for s in ("l", "r")])
    masks = []
    for start in a.window_starts:
        end = start+a.window_seconds
        mask = (d["time_s"] >= start) & (d["time_s"] < end)
        if (mask.sum() < round(a.window_seconds/dt)-1 or d["done"][mask].any()
                or len(set(d["episode_id"][mask])) != 1
                or not np.allclose(d["command_applied"][mask], a.vx, atol=1e-6, rtol=0)):
            raise ValueError("Requested analysis window crosses a reset, buffer or command change")
        result = describe(q[mask], local[mask], sole[mask], d["base_vel_x"][mask], names, dt, force[mask])
        result["time_window_s"] = [start, end]
        result["mean_base_pitch_deg"] = float(np.rad2deg(d["base_euler_y"][mask].mean()))
        report["policy_windows"]["{:g}-{:g}".format(start, end)] = result
        masks.append(mask)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/"forward_gait_analysis.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    plot(a.output_dir, d["time_s"][masks[0]], q[masks[0]], sole[masks[0]], names, time, rq.numpy(), rs, report)
    print(json.dumps({"foot_fk_max_error_m": fk_error, "reference_frequency_hz": rate,
                      "windows": {k: {"mean_vx": v["mean_vx"], "oscillation_hz": v["dominant_hip_knee_oscillation_hz"],
                                      "feet": v["feet"]} for k, v in report["policy_windows"].items()}}))


def plot(output, actual_time, q, sole, names, time, rq, rs, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    speed = "{:+g}".format(report["command_vx"])
    for i, side in enumerate(("left", "right")):
        idx = names.index(side+"_knee_pitch_joint")
        axes[0, i].plot(time, np.rad2deg(rq[:, idx]), label="Reference at "+speed+" m/s", color="#2a9377")
        axes[0, i].plot(actual_time-actual_time[0], np.rad2deg(q[:, idx]), label="Policy", color="#b34b69")
        axes[0, i].set(title=side.title()+" knee pitch", ylabel="Degrees", xlabel="Seconds", xlim=(0, 6))
        axes[0, i].legend()
        axes[1, i].plot(time, rs[:, i]*1000, label="Kinematic reference", color="#2a9377")
        axes[1, i].plot(actual_time-actual_time[0], sole[:, i]*1000, label="Actual minimum sole", color="#b34b69")
        axes[1, i].axhline(10, color="gray", linestyle="--", linewidth=.8)
        axes[1, i].set(title=side.title()+" minimum sole clearance", ylabel="mm", xlabel="Seconds", xlim=(0, 6))
        axes[1, i].legend()
    selected = next(iter(report["policy_windows"].values()))
    x = np.arange(2)
    for offset, item, color, label in ((-.18, report["reference"], "#2a9377", "Reference"), (.18, selected, "#b34b69", "Policy")):
        axes[2, 0].bar(x+offset, [item["feet"][s]["forward_excursion_p95_p05_m"]*100 for s in ("left", "right")], .36, color=color, label=label)
    axes[2, 0].set(xticks=x, xticklabels=["Left", "Right"], ylabel="cm (P95-P05)", title="Foot fore-aft excursion relative to pelvis")
    axes[2, 0].legend()
    axes[2, 1].bar(["Reference phase", "Policy joint oscillation"], [report["reference_phase_rate_hz"], selected["dominant_hip_knee_oscillation_hz"]], color=["#2a9377", "#b34b69"])
    axes[2, 1].set(ylabel="Hz", title="Frequency comparison (not touchdown step count)")
    start, end = selected["time_window_s"]
    fig.suptitle("Forward gait: requested "+speed+" m/s; first continuous window {:g}-{:g} s\nReference follows checkpoint semantics; sole reference is kinematic".format(start, end), fontsize=13)
    fig.savefig(output/"forward_gait_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
