"""Inspect a saved AMP discriminator using reconstructed, verified CSV histories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.models import Discriminator
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.stationary import CommandMotionSampler


def analyze(model, csv_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    names = metadata["dof_names"]
    count, history = len(names), metadata["history_steps"]
    library = MotionLibrary(
        ROOT / "resources/motions/slow_walk_amp_v1.2/normalized",
        ROOT / "resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf",
        names, metadata["key_body_names"], metadata["policy_dt"], history,
        ground_clearance=.002)
    disc = Discriminator(library.feature_dim, history,
                         checkpoint["config"]["discriminator"]["hidden_dims"])
    disc.load_state_dict(checkpoint["discriminator_state_dict"])
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    def tensor(keys):
        return torch.tensor(np.column_stack([data[k] for k in keys]), dtype=torch.float32)

    root_keys = ["x", "y", "z", "qx", "qy", "qz", "qw",
                 "world_vx", "world_vy", "world_vz", "world_wx", "world_wy", "world_wz"]
    states = torch.cat([tensor(root_keys), tensor(["q_"+n for n in names]),
                        tensor(["dq_"+n for n in names])], -1)
    if not torch.isfinite(states).all():
        raise ValueError("Nonfinite replay states")
    features = library.features(states)
    indices = np.arange(history-1, len(data))
    offsets = np.arange(history-1, -1, -1)
    histories = features[torch.tensor(indices[:, None]-offsets[None, :])]
    # Exclude reset histories and command/buffer transitions from diagnosis.
    events = np.ones(len(data), dtype=bool)
    events[1:] = ((np.abs(np.diff(data["command_applied"])) > 1e-7)
                  | (np.diff(data["episode_id"]) != 0)
                  | (np.diff(data["active_stage"]) != 0))
    if "stand_steps_left" in data.dtype.names:
        events[1:] |= ((data["stand_steps_left"][:-1] > 0) & (data["stand_steps_left"][1:] == 0))
    last_event = np.maximum.accumulate(np.where(events, data["time_s"], -np.inf))
    valid = ((data["episode_step"][indices] > 100) & (data["z"][indices] > .45)
             & ~data["done"][indices].astype(bool)
             & (data["time_s"][indices]-last_event[indices] >= 2.-1e-6))
    if "stand_steps_left" in data.dtype.names:
        valid &= data["stand_steps_left"][indices] == 0
    indices, histories = indices[valid], histories[valid]
    if not len(indices):
        raise ValueError("No settled replay samples")
    with torch.no_grad():
        reconstructed = disc.style_reward(histories).numpy()
    error = float(np.max(np.abs(reconstructed-data["style_score"][indices])))
    if error > .005:
        raise ValueError("Reconstructed AMP history differs from recorded score: " + str(error))
    sampler = CommandMotionSampler(library)
    ids, times = sampler.sample_matching(tensor(["cmd_vx", "cmd_vy", "cmd_wz"])[indices])
    expert = library.history(ids, times)
    groups = {"height": [0], "gravity": list(range(1, 4)),
              "linear_velocity": list(range(4, 7)), "angular_velocity": list(range(7, 10)),
              "joint_position": list(range(10, 10+count)),
              "joint_velocity": list(range(10+count, 10+2*count)),
              "key_positions": list(range(10+2*count, library.feature_dim))}
    result = {}
    for label, mask in (("standing", np.abs(data["command_applied"][indices]) < 1e-6),
                        ("walking", np.abs(data["command_applied"][indices]) >= 1e-6)):
        if not mask.any():
            continue
        policy, reference = histories[mask], expert[mask]
        normalized = disc.normalizer(policy).detach().requires_grad_(True)
        score = disc.network(normalized.flatten(1)).squeeze(-1)
        gradient = torch.autograd.grad(score.sum(), normalized)[0]
        item = {"samples": len(policy), "policy_D_mean": float(score.mean().detach()),
                "policy_style_mean": float(disc.style_reward(policy).mean()),
                "matched_expert_D_mean": float(disc(reference).mean().detach()), "groups": {}}
        with torch.no_grad():
            for name, columns in groups.items():
                hybrid = policy.clone()
                hybrid[:, :, columns] = reference[:, :, columns]
                item["groups"][name] = {
                    "normalized_input_mean_abs_gradient": float(gradient[:, :, columns].abs().mean()),
                    "replace_group_delta_D": float((disc(hybrid)-score.detach()).mean()),
                    "mean_abs_standardized_difference": float(
                        (disc.normalizer(policy)[:, :, columns]-disc.normalizer(reference)[:, :, columns]).abs().mean())}
        result[label] = item
    return {"iteration": checkpoint["iter"], "seed": 5, "style_reconstruction_max_error": error,
            "method": "Matched real reference histories; frozen discriminator and normalizer. Only settled replay samples.",
            "limits": "Feature substitution can be physically inconsistent. Delta D and local gradients describe discriminator sensitivity; they do not prove a training fix or gait causality.",
            "conditions": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.model, args.csv)
    result.update(model=args.model.as_posix(), csv=args.csv.as_posix())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "amp_feature_analysis.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
