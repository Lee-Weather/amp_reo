"""Measure the actual command-matched RSI reference pools without running physics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.stationary import CommandMotionSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = state["metadata"]
    lib = MotionLibrary(ROOT/"resources/motions/slow_walk_amp_v1.2/normalized",
                        ROOT/"resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf",
                        meta["dof_names"], meta["key_body_names"], meta["policy_dt"],
                        meta["history_steps"], ground_clearance=.002)
    sampler = CommandMotionSampler(lib, .15, .2, .05, .3, .2)
    prior = sampler.prior.numpy()
    quiet = sampler.quiet.numpy()
    vx = sampler.local_vx.numpy()
    ids, times = sampler.ids.numpy(), sampler.times.numpy()
    expert_mass = .8*prior/prior.sum()+.2*prior*quiet/(prior*quiet).sum()
    clips = {}
    for index, name in enumerate(lib.names):
        mask = ids == index
        clips[name] = dict(candidate_windows=int(mask.sum()), quiet_windows=int(quiet[mask].sum()),
                           duration_s=float(lib.durations[index]),
                           local_vx_quantiles=np.quantile(vx[mask], [0, .05, .5, .95, 1]).tolist())
    commands = {}
    for command in [-.2, -.1, 0., .1, .2, .4, .6, .65]:
        allowed = (abs(vx-command) <= sampler.tolerance) & ((abs(command) <= .05) | (command*vx > 0))
        if command == 0:
            allowed = quiet.copy()
        weights = prior*allowed
        if weights.sum() <= 0:
            raise ValueError("No reference support for command "+str(command))
        weights /= weights.sum()
        per_clip = {}
        for index, name in enumerate(lib.names):
            eligible = allowed[ids == index]
            segments = np.diff(np.r_[0, eligible.astype(int), 0])
            starts, ends = np.flatnonzero(segments == 1), np.flatnonzero(segments == -1)
            # Duration covered by consecutive candidate endpoint samples;
            # a single endpoint has zero sustained duration.
            longest = max((int(b-a)-1 for a, b in zip(starts, ends)), default=0)/float(lib.fps[index])
            per_clip[name] = dict(probability=float(weights[ids == index].sum()),
                                  eligible_windows=int(eligible.sum()), longest_endpoint_run_s=longest)
        commands[str(command)] = dict(candidate_windows=int(allowed.sum()),
                                     weighted_mean_reference_vx=float((weights*vx).sum()),
                                     quiet_probability=float(weights[quiet].sum()),
                                     root_abs_vx_below_005_probability=float(weights[abs(vx) < .05].sum()),
                                     expert_mixture_matching_speed_mass=float(expert_mass[allowed].sum()),
                                     clips=per_clip)
    report = dict(source_model=args.model.as_posix(), source_iteration=int(state["iter"]),
                  scope="Reference distribution audit only; no physics, training or policy success inference",
                  settings=dict(speed_tolerance=.15, expert_standing_probability=.2,
                                quiet_root_speed=.05, quiet_angular_speed=.3, quiet_joint_speed=.2,
                                clip_prior="Equal clip weights, same as exp1.4 through exp1.6"),
                  approximation="Reference grid endpoints approximate continuous expert time sampling; RSI discrete endpoint probabilities are exact for this sampler.",
                  clips=clips, commands=commands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:{m:v for m,v in value.items() if m != "clips"}
                      for k,value in commands.items()}))


if __name__ == "__main__":
    main()
