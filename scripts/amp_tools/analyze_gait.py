"""Compare learned gait and trusted references without simulator or policy edits."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks, periodogram
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.motion import URDFKinematics, quat_mul, quat_rotate, quat_inverse_rotate
from humanoid.amp.grounding import FootGrounding


def span(array):
    return float(np.quantile(array, .95)-np.quantile(array, .05))


def frequency(array, dt):
    frequencies, power = periodogram(array, fs=1/dt, detrend="linear")
    selected = (frequencies >= .3) & (frequencies <= 8.)
    dominant = float(frequencies[selected][np.argmax(power[selected])])
    peaks, _ = find_peaks(array, distance=max(1, round(.12/dt)), prominence=max(1e-5, .25*span(array)))
    period = float(np.median(np.diff(peaks))*dt) if len(peaks) >= 3 else None
    return dict(dominant_hz=dominant, peak_count=len(peaks), median_period_s=period,
                cycle_hz=1/period if period else dominant)


def geometry(root, q, fk, grounding):
    position, orientation = fk.key_transforms(q)
    world = root[:, None, :3]+quat_rotate(root[:, None, 3:7].expand(-1, 2, -1), position)
    minima = []
    for index, vertices in grounding.feet:
        rotation = quat_mul(root[:, 3:7], orientation[:, index])
        x, y, z, w = rotation.unbind(-1)
        zrow = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
        minima.append((zrow @ vertices.T).min(-1).values+world[:, index, 2])
    return position.numpy(), world.numpy(), torch.stack(minima, -1).numpy()


def metrics(q, local_feet, soles, vx, names, dt, forces=None):
    output = dict(samples=len(q), duration_s=(len(q)-1)*dt, mean_vx=float(vx.mean()), joints={}, feet={})
    for side in ("left", "right"):
        for joint in ("hip_pitch", "knee_pitch", "ankle_pitch"):
            name = side+"_"+joint+"_joint"
            signal = q[:, names.index(name)]
            output["joints"][name] = dict(mean_deg=float(np.rad2deg(signal.mean())),
                                          p95_p05_deg=float(np.rad2deg(span(signal))),
                                          frequency=frequency(signal, dt))
    for i, side in enumerate(("left", "right")):
        output["feet"][side] = dict(sole_p05_m=float(np.quantile(soles[:, i], .05)),
                                    sole_p95_m=float(np.quantile(soles[:, i], .95)),
                                    sole_vertical_span_m=span(soles[:, i]),
                                    sole_below_1cm_fraction=float((soles[:, i] <= .01).mean()),
                                    forward_excursion_p95_p05_m=span(local_feet[:, i, 0]))
        if forces is not None:
            output["feet"][side]["contact_over_20N_fraction"] = float((forces[:, i] > 20).mean())
    output["geometric_double_support_fraction"] = float((soles <= .01).all(-1).mean())
    if forces is not None:
        output["force_double_support_fraction"] = float((forces > 20).all(-1).mean())
    periods = [output["joints"][s+"_hip_pitch_joint"]["frequency"]["median_period_s"]
               for s in ("left", "right")]
    period = float(np.median([p for p in periods if p is not None]))
    output["same_leg_cycle_period_s"] = period
    output["trunk_forward_distance_per_same_leg_cycle_m"] = float(vx.mean()*period)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    names = checkpoint["metadata"]["dof_names"]
    dt = checkpoint["metadata"]["policy_dt"]
    urdf = ROOT / "resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
    feet_names = [s+"_ankle_roll_link" for s in ("left", "right")]
    fk = URDFKinematics(urdf, names, feet_names)
    grounding = FootGrounding(urdf, fk, .002)
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    root = torch.tensor(np.column_stack([data[k] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]), dtype=torch.float32)
    q = np.column_stack([data["q_"+n] for n in names])
    local, world, sole = geometry(root, torch.tensor(q, dtype=torch.float32), fk, grounding)
    # Confirm that reconstructed link origins agree with simulator diagnostics.
    logged = np.stack([np.column_stack([data["foot_"+axis+"_"+s] for axis in "xyz"]) for s in ("l", "r")], 1)
    # Rigid-body transforms in the CSV can lag reset tensor writes on done rows.
    stable = (data["episode_step"] > 2) & ~data["done"].astype(bool)
    fk_error = float(np.max(np.abs(world[stable]-logged[stable])))
    if fk_error > 1e-4:
        raise ValueError("Foot geometry does not match CSV: "+str(fk_error))
    force = np.column_stack([data["foot_force_"+s] for s in ("l", "r")])
    report = dict(definitions={
        "frequency": "Same-leg joint oscillation, NOT bilateral touchdown step count",
        "excursion": "P95-P05; foot position relative to pelvis",
        "sole_height": "Lowest raw collision-mesh vertex, reconstructed by URDF FK",
        "cycle_distance": "Mean body-forward velocity times same-leg period; not measured touchdown step length",
        "reference": "B3 forward continuous gait; boundary frames excluded; unequal actual speed is reported",
    }, foot_fk_max_error_m=fk_error, policy_windows={}, reference_clips={})
    for label, low, high in (("cmd04_first", 12., 19.8), ("cmd06", 23.6, 29.8), ("cmd04_second", 43.6, 49.8)):
        mask = (data["time_s"] >= low) & (data["time_s"] <= high)
        result = metrics(q[mask], local[mask], sole[mask], data["base_vel_x"][mask], names, dt, force[mask])
        result["time_window_s"] = [low, high]
        result["actuation"] = {}
        for name in ("left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_pitch_joint",
                     "right_knee_pitch_joint", "lumbar_pitch_joint", "right_hip_roll_joint"):
            signals = {}
            for prefix in ("pd_target_raw_", "pd_target_effective_", "q_", "torque_"):
                signal = data[prefix+name][mask]
                angular = prefix != "torque_"
                signals[prefix.rstrip("_")] = dict(
                    mean=float(np.rad2deg(signal.mean()) if angular else signal.mean()),
                    p95_p05=float(np.rad2deg(span(signal)) if angular else span(signal)),
                    unit="deg" if angular else "Nm", frequency=frequency(signal, dt))
            result["actuation"][name] = signals
        report["policy_windows"][label] = result
    reference_series = None
    for path in sorted((ROOT / "resources/motions/slow_walk_amp_v1.2/normalized").glob("*.pkl")):
        with path.open("rb") as stream:
            clip = pickle.load(stream)
        rq = torch.tensor(clip["dof_pos"][:, [clip["dof_names"].index(n) for n in names]], dtype=torch.float32)
        rq = rq.maximum(fk.lower).minimum(fk.upper)
        rot = torch.tensor(clip["root_rot"], dtype=torch.float32)
        rot = rot/rot.norm(dim=-1, keepdim=True)
        pos = grounding.align(torch.tensor(clip["root_pos"], dtype=torch.float32), rot, rq)
        rroot = torch.cat((pos, rot), -1)
        rl, rw, rs = geometry(rroot, rq, fk, grounding)
        rdt = 1/float(clip["fps"])
        velocity = torch.tensor(np.gradient(pos.numpy(), rdt, axis=0))
        vx = quat_inverse_rotate(rot, velocity).numpy()[:, 0]
        time = np.arange(len(rq))*rdt
        mask = (time >= .5) & (time <= time[-1]-.5)
        report["reference_clips"][path.stem] = metrics(rq.numpy()[mask], rl[mask], rs[mask], vx[mask], names, rdt)
        if path.stem == "F1_B3_continuous_walk":
            reference_series = (time[mask]-time[mask][0], rq.numpy()[mask], rl[mask], rs[mask])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"gait_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_comparison(data, q, local, sole, names, reference_series, report, args.output_dir)
    print(json.dumps(dict(foot_fk_error_m=fk_error, policy=report["policy_windows"],
                          reference_B3=report["reference_clips"]["F1_B3_continuous_walk"])))


def plot_comparison(data, q, local, soles, names, reference, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if "Microsoft YaHei" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    mask = (data["time_s"] >= 23.6) & (data["time_s"] <= 29.8)
    time = data["time_s"][mask]-data["time_s"][mask][0]
    rt, rq, rl, rs = reference
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for i, side in enumerate(("左", "右")):
        axes[0, i].plot(rt, rs[:, i]*100, label="参考B3", color="#2a9377")
        axes[0, i].plot(time, soles[mask, i]*100, label="策略 cmd=0.6", color="#b34b69")
        axes[0, i].set(title=side+"脚最低碰撞脚底离地高度", xlabel="秒", ylabel="厘米", xlim=(0, 6.2))
        axes[0, i].legend()
    for ax, name, title in ((axes[1, 0], "left_hip_pitch_joint", "左髋pitch"), (axes[1, 1], "left_knee_pitch_joint", "左膝pitch")):
        index = names.index(name)
        ax.plot(rt, np.rad2deg(rq[:, index]), label="参考B3", color="#2a9377")
        ax.plot(time, np.rad2deg(data["pd_target_effective_"+name][mask]),
                label="有效PD目标", color="#da9b31", alpha=.65, linewidth=.9)
        ax.plot(time, np.rad2deg(q[mask, index]), label="策略 cmd=0.6", color="#b34b69")
        ax.set(title=title+"：频率与摆幅", xlabel="秒", ylabel="度", xlim=(0, 6.2))
        ax.legend()
    for i, side in enumerate(("left", "right")):
        axes[2, 0].bar(i-.18, report["reference_clips"]["F1_B3_continuous_walk"]["feet"][side]["forward_excursion_p95_p05_m"]*100, .35, color="#2a9377", label="参考B3" if i == 0 else None)
        axes[2, 0].bar(i+.18, report["policy_windows"]["cmd06"]["feet"][side]["forward_excursion_p95_p05_m"]*100, .35, color="#b34b69", label="策略" if i == 0 else None)
    axes[2, 0].set(xticks=[0, 1], xticklabels=["左脚", "右脚"], ylabel="厘米", title="脚相对骨盆前后摆幅（P95-P05）")
    axes[2, 0].legend()
    stages = ["B3参考", "策略0.4指令", "策略0.6指令"]
    items = [report["reference_clips"]["F1_B3_continuous_walk"], report["policy_windows"]["cmd04_first"], report["policy_windows"]["cmd06"]]
    axes[2, 1].bar(stages, [1/r["same_leg_cycle_period_s"] for r in items], color=["#2a9377", "#b34b69", "#b34b69"])
    axes[2, 1].set(ylabel="Hz（同腿周期）", title="同腿关节运动周期")
    suffix = "_evaluation"
    experiment = output.name[:-len(suffix)] if output.name.endswith(suffix) else output.name
    fig.suptitle(experiment+" 步态诊断\n参考B3与0.6m/s指令段对比；实际速度不同，详见数值报告", fontsize=15)
    fig.savefig(output/"gait_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
