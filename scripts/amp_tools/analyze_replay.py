"""Shared, explicit metrics for AMP CSVs; no simulator or training side effects."""
import csv
from pathlib import Path
import pickle

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_context(model):
    state = torch.load(model, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    names = metadata["dof_names"]
    reference = []
    for path in sorted((ROOT / "resources/motions/slow_walk_amp_v1.2/normalized").glob("*.pkl")):
        with path.open("rb") as stream:
            clip = pickle.load(stream)
        order = [clip["dof_names"].index(name) for name in names]
        positions = clip["dof_pos"][:, order]
        bounds = metadata.get("control", {}).get("hard_limits")
        if bounds is not None:
            bounds = np.asarray(bounds)
            positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        reference.append(positions)
    return dict(metadata=metadata, names=names, iteration=state["iter"],
                style_reward_scale=state.get("config", {}).get("discriminator", {}).get("style_reward_scale"),
                reference_mean=np.concatenate(reference).mean(0),
                default=np.asarray(metadata["default_dof_pos"]), physics_dt=state["config"]["sim"]["dt"])


def evaluate_csv(path, context):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty replay CSV")
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError("Nonfinite replay CSV")
    names, dt = context["names"], context["metadata"]["policy_dt"]
    done, timeout = data["done"].astype(bool), data["timeout"].astype(bool)
    time, command = data["time_s"], data["command_applied"]
    observed_error = float(np.max(np.abs(command-data["command_observed"])))
    if observed_error > 1e-6:
        raise ValueError("Command/observation mismatch")
    # Wait two seconds after reset, applied-command changes, or buffer expiry.
    # These are transitions in simulated time, never wall-clock time.
    events = np.zeros(len(rows), dtype=bool)
    events[0] = True
    events[1:] = ((np.abs(np.diff(command)) > 1e-7)
                  | (np.diff(data["episode_id"]) != 0)
                  | (np.diff(data["active_stage"]) != 0))
    if "stand_steps_left" in data:
        events[1:] |= ((data["stand_steps_left"][:-1] > 0)
                       & (data["stand_steps_left"][1:] == 0))
    last_event = np.maximum.accumulate(np.where(events, time, -np.inf))
    settled = (time-last_event >= 2.-1e-6) & ~done
    if "stand_steps_left" in data:
        settled &= data["stand_steps_left"] == 0
    stable = (data["episode_step"] > 100) & ~done & (data["base_height"] > .45)

    def speed_stats(mask):
        if not mask.any():
            return dict(samples=0)
        velocity, target = data["base_vel_x"][mask], command[mask]
        return dict(samples=int(mask.sum()), mean_command=float(target.mean()),
                    mean_vx=float(velocity.mean()), mean_abs_vx=float(np.abs(velocity).mean()),
                    rmse_vx=float(np.sqrt(np.mean((velocity-target)**2))),
                    mean_abs_vy=float(np.abs(data["base_vel_y"][mask]).mean()),
                    mean_abs_yaw_rate=float(np.abs(data["base_ang_vel_z"][mask]).mean()))

    segments = []
    for stage in sorted(np.unique(data["active_stage"]).astype(int)):
        mask = data["active_stage"] == stage
        segments.append(dict(stage=int(stage), requested=float(data["command_requested"][mask][0]),
                             resets=int((mask & done).sum()), timeouts=int((mask & done & timeout).sum()),
                             all_steps=speed_stats(mask), settled=speed_stats(mask & settled)))
    standing = settled & (np.abs(data["command_requested"]) < 1e-7)
    walking = settled & (np.abs(data["command_requested"]) > 1e-7)
    positive04 = settled & np.isclose(data["command_requested"], .4)
    positive06 = settled & np.isclose(data["command_requested"], .6)
    q = np.column_stack([data["q_"+name] for name in names])
    reference, default = context["reference_mean"], context["default"]
    joint_stats = {}
    decimation = round(dt/context["physics_dt"])
    bounds = context["metadata"].get("control", {}).get("hard_limits")
    for i, name in enumerate(names):
        mask = stable
        raw = data.get("pd_target_raw_"+name, data["target_"+name])
        effective = data.get("pd_target_effective_"+name, data["target_"+name])
        clip = data.get("pd_target_clip_counts_"+name)
        joint = dict(reference_mean_deg=float(np.rad2deg(reference[i])),
                     stable_mean_deg=float(np.rad2deg(q[mask, i].mean())) if mask.any() else None,
                     stable_std_deg=float(np.rad2deg(q[mask, i].std())) if mask.any() else None,
                     stable_mean_offset_deg=float(np.rad2deg(q[mask, i].mean()-reference[i])) if mask.any() else None,
                     raw_clip_rate=float(clip.mean()/decimation) if clip is not None else None)
        if bounds is not None:
            lower, upper = bounds[i]
            joint.update(effective_target_violation_rate=float(((effective < lower-1e-6) | (effective > upper+1e-6)).mean()),
                         maximum_raw_violation_rad=float(np.maximum(np.maximum(lower-raw, raw-upper), 0).max()))
        joint_stats[name] = joint
    groups = {
        "waist": [i for i, n in enumerate(names) if n.startswith("lumbar_")],
        "upper": [i for i, n in enumerate(names) if any(t in n for t in ("shoulder", "elbow", "wrist"))],
        "leg_lateral": [i for i, n in enumerate(names) if any(t in n for t in ("hip_roll", "hip_yaw", "ankle_roll"))],
    }
    posture = {key: float(np.rad2deg(np.sqrt(np.mean((q[stable][:, ids]-default[ids])**2))))
               if stable.any() else None for key, ids in groups.items()}
    failures = [dict(time_s=float(time[i]), episode_step=int(data["episode_step"][i]),
                     requested=float(data["command_requested"][i]), applied=float(command[i]),
                     height=float(data["base_height"][i]), roll_deg=float(np.rad2deg(data["base_euler_x"][i])),
                     pitch_deg=float(np.rad2deg(data["base_euler_y"][i])))
                for i in np.where(done & ~timeout)[0]]
    mean_task, mean_style = float(data["task_reward"].mean()), float(data["style_reward"].mean())
    clip_rates = [v["raw_clip_rate"] for v in joint_stats.values() if v["raw_clip_rate"] is not None]
    report = dict(rows=len(rows), duration_s=float(time[-1]), control_dt=dt,
                  resets=int(done.sum()), timeouts=int((done & timeout).sum()),
                  non_timeout_terminations=int((done & ~timeout).sum()), failures=failures,
                  command_observation_max_error=observed_error, stable_samples=int(stable.sum()),
                  settled_samples=int(settled.sum()), mean_task_reward=mean_task, mean_style_reward=mean_style,
                  style_numerical_share=mean_style/(mean_task+mean_style) if mean_task+mean_style else None,
                  style_zero_fraction=float((data["style_reward"] <= 1e-9).mean()),
                  posture_group_rms_deg=posture, joints=joint_stats, segments=segments,
                  standing=speed_stats(standing), walking=speed_stats(walking),
                  speed_04=speed_stats(positive04), speed_06=speed_stats(positive06),
                  raw_target_clip_rate=float(np.mean(clip_rates)) if clip_rates else None,
                  torque_saturation_rate=float(data["clip_count"][-1]/(len(rows)*len(names)*decimation)),
                  stable_minimum_height=float(data["base_height"][stable].min()) if stable.any() else None)
    report["criteria"] = {
        "no_non_timeout_termination": report["non_timeout_terminations"] == 0,
        "effective_targets_within_hard_limits": all(v["effective_target_violation_rate"] == 0 for v in joint_stats.values()) if bounds is not None else None,
        "standing_mean_abs_vx_below_005": report["standing"].get("samples", 0) > 0 and report["standing"]["mean_abs_vx"] <= .05,
        "speed_04_rmse_below_015": report["speed_04"].get("samples", 0) > 0 and report["speed_04"]["rmse_vx"] <= .15,
        "speed_06_rmse_below_015": report["speed_06"].get("samples", 0) > 0 and report["speed_06"]["rmse_vx"] <= .15,
        # The .005 threshold was set for exp_1.1's scale=2.  It is not a
        # valid pass/fail threshold for differently scaled AMP rewards.
        "style_reward_above_0005": (mean_style >= .005) if context.get("style_reward_scale") == 2. else None,
    }
    if context.get("style_reward_scale") != 2.:
        report["style_criterion_note"] = "0.005 threshold applies to scale=2; judge gait separately for other scales."
    return report
