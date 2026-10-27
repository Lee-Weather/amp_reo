"""Measure persistent zero-command motion in the six-stage seed=5 replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def span(signal):
    return float(np.quantile(signal, .95) - np.quantile(signal, .05))


def analyze(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x, y, z, w = (data[name] for name in ("qx", "qy", "qz", "qw"))
    roll = np.rad2deg(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    windows = {}
    masks = []
    for label, start, end in (("initial_stop", 3.6, 9.8),
                              ("after_first_walk", 34., 39.8),
                              ("after_second_walk", 54., 59.8)):
        mask = (data["time_s"] >= start) & (data["time_s"] < end)
        if not mask.any() or data["done"][mask].any():
            raise ValueError("Standing window is empty or includes a reset: " + label)
        command = np.column_stack([data[k][mask] for k in ("cmd_vx", "cmd_vy", "cmd_wz",
                                                         "cmd_linear_x", "cmd_linear_y", "cmd_angular_z")])
        if not np.allclose(command, 0, atol=1e-8):
            raise ValueError("Standing window includes nonzero commands: " + label)
        joints = {}
        for side in ("left", "right"):
            for joint in ("knee_pitch", "ankle_pitch"):
                name = side + "_" + joint + "_joint"
                joints[name] = {
                    "mean_q_deg": float(np.rad2deg(data["q_"+name][mask].mean())),
                    "q_p95_p05_deg": float(np.rad2deg(span(data["q_"+name][mask]))),
                    "dq_rms_rad_s": float(np.sqrt(np.mean(data["dq_"+name][mask]**2))),
                    "raw_target_p95_p05_deg": float(np.rad2deg(span(data["pd_target_raw_"+name][mask]))),
                    "effective_target_p95_p05_deg": float(np.rad2deg(span(data["pd_target_effective_"+name][mask]))),
                }
        windows[label] = {
            "requested_window_s": [start, end], "rows": int(mask.sum()),
            "max_abs_command": float(np.abs(command).max()),
            "mean_abs_vx_m_s": float(np.abs(data["vx"][mask]).mean()),
            "mean_abs_vy_m_s": float(np.abs(data["vy"][mask]).mean()),
            "mean_abs_wz_rad_s": float(np.abs(data["wz"][mask]).mean()),
            "roll_mean_deg": float(roll[mask].mean()), "roll_p95_p05_deg": span(roll[mask]),
            "pitch_mean_deg": float(pitch[mask].mean()), "pitch_p95_p05_deg": span(pitch[mask]),
            "root_height_p95_p05_mm": 1000*span(data["z"][mask]),
            "yaw_change_deg": float(np.rad2deg(yaw[mask][-1]-yaw[mask][0])),
            "left_contact_over_20N_fraction": float((data["foot_force_l"][mask] > 20).mean()),
            "right_contact_over_20N_fraction": float((data["foot_force_r"][mask] > 20).mean()),
            "joints": joints,
        }
        masks.append(mask)
    return data, roll, pitch, masks, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Defaults to output_dir/replay_metadata.json")
    args = parser.parse_args()
    metadata_path = args.metadata or args.output_dir/"replay_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["seed"] != 5:
        raise ValueError("This project's standing analysis uses seed=5")
    data, roll, pitch, masks, windows = analyze(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"csv": args.csv.as_posix(), "metadata_source": metadata_path.as_posix(),
              "experiment": args.output_dir.name.removesuffix("_evaluation"), "seed": metadata["seed"],
              "replay_conditions": {key: metadata[key] for key in ("seed", "noise", "pushes", "reset_mode")},
              "definitions": {"amplitude": "P95-P05, not peak-to-peak or mean posture offset",
                              "windows": "Late zero-command windows; initial window precedes any walking",
                              "height": "Root height, not whole-body COM height",
                              "contact": "CSV foot contact vertical force >20 N"},
              "windows": windows}
    (args.output_dir/"standing_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex="col")
    for col, (label, mask) in enumerate(zip(windows, masks)):
        t = data["time_s"][mask]
        for name in ("vx", "vy"):
            axes[0, col].plot(t, data[name][mask], label=name)
        axes[0, col].axhline(0, color="black", lw=.5)
        for name, signal in (("roll", roll), ("pitch", pitch)):
            axes[1, col].plot(t, signal[mask], label=name)
        for side in ("left", "right"):
            name = side+"_knee_pitch_joint"
            axes[2, col].plot(t, np.rad2deg(data["q_"+name][mask]), label=side)
            axes[3, col].plot(t, np.rad2deg(data["pd_target_effective_"+name][mask]), label=side)
        axes[0, col].set_title(label.replace("_", " "))
        axes[3, col].set_xlabel("Replay time (s)")
        for row in range(4):
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend(fontsize=8)
    for row, label in enumerate(("Body velocity (m/s)", "Root orientation (deg)",
                                 "Actual knee angle (deg)", "Applied knee PD target (deg)")):
        axes[row, 0].set_ylabel(label)
    fig.suptitle(args.output_dir.name.replace("_evaluation", "")+": sustained motion at zero command (seed=5)")
    fig.tight_layout()
    fig.savefig(args.output_dir/"standing_analysis.png", dpi=160)
    plt.close(fig)
    print(json.dumps(windows, ensure_ascii=False))


if __name__ == "__main__":
    main()
