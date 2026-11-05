"""Opt-in, pickle-free traces for matched-state, matched-action engine probes.

Rows contain the observation/action BEFORE a control step and physical state
AFTER it, before any automatic reset. Forces are diagnostic, not an equivalence
claim about the two contact solvers.
"""
import json
from pathlib import Path

import numpy as np


SCHEMA = "x1_transition_trace_v1"


class TransitionTrace:
    def __init__(self, metadata, root, q, dq, observation):
        self.metadata = dict(metadata, schema=SCHEMA)
        self.initial = {"initial_root": np.asarray(root).copy(),
                        "initial_q": np.asarray(q).copy(),
                        "initial_dq": np.asarray(dq).copy(),
                        "initial_observation": np.asarray(observation).copy()}
        self.rows = []

    def append(self, observation, action, root, q, dq, forces, done=False):
        if self.rows and self.rows[-1]["done"]:
            raise ValueError("A transition trace must stop at the first reset")
        self.rows.append({key: np.asarray(value).copy() for key, value in dict(
            observations=observation, actions=action, roots=root, q=q, dq=dq,
            forces=forces, done=bool(done)).items()})

    def arrays(self):
        if not self.rows:
            raise ValueError("Cannot save an empty transition trace")
        data = dict(self.initial)
        data.update({key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
        data["time_s"] = np.arange(1, len(self.rows)+1)*self.metadata["policy_dt"]
        data["metadata"] = np.array(json.dumps(self.metadata, allow_nan=False))
        validate_trace(data)
        return data

    def save(self, path):
        data = self.arrays()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)


def validate_trace(data):
    meta = json.loads(str(data["metadata"]))
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported transition trace schema")
    names = meta["dof_names"]
    n, width = len(names), meta["num_observations"]
    if not n or len(set(names)) != n or width < 1:
        raise ValueError("Invalid joint names or observation dimension")
    dt = float(meta["policy_dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control period")
    count = len(data["actions"])
    shapes = {"initial_root": (13,), "initial_q": (n,), "initial_dq": (n,),
              "initial_observation": (width,), "observations": (count, width),
              "actions": (count, n), "roots": (count, 13), "q": (count, n),
              "dq": (count, n), "forces": (count, 2, 3), "done": (count,),
              "time_s": (count,)}
    if not count:
        raise ValueError("Empty transition trace")
    for key, shape in shapes.items():
        value = data[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid trace array: "+key)
    if (not np.isin(data["done"], [0, 1]).all() or data["done"][:-1].any()
            or not np.allclose(data["time_s"], np.arange(1, count+1)*dt, rtol=0, atol=1e-7)):
        raise ValueError("Trace contains a reset boundary or a discontinuous clock")
    quats = np.vstack((data["initial_root"][3:7], data["roots"][:, 3:7]))
    if not np.allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-3, rtol=0):
        raise ValueError("Trace root quaternion must be unit xyzw")
    if not np.array_equal(data["initial_observation"], data["observations"][0]):
        raise ValueError("First observation differs from the captured initial history")
    return meta


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_trace(data)
    return data


def validate_replay_source(data, checkpoint_sha256, dof_names, policy_dt, num_observations):
    meta = validate_trace(data)
    expected = dict(checkpoint_sha256=checkpoint_sha256, dof_names=list(dof_names),
                    num_observations=num_observations)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Action trace belongs to a different checkpoint, joint order or policy")
    if abs(meta["policy_dt"]-policy_dt) > 1e-7:
        raise ValueError("Action trace control period differs")
    # Moving root velocities need an independently verified origin/COM mapping.
    # This first probe deliberately supports exact, stationary startup states.
    if (np.max(np.abs(data["initial_root"][7:])) > 1e-7
            or np.max(np.abs(data["initial_dq"])) > 1e-7):
        raise ValueError("Matched-action replay currently requires a stationary initial state")
    return meta


def compare_traces(source, target, position_tolerance=.01, orientation_tolerance=.05,
                   joint_tolerance=.05):
    left, right = validate_trace(source), validate_trace(target)
    for key in ("checkpoint_sha256", "dof_names", "num_observations"):
        if left[key] != right[key]:
            raise ValueError("Trace identity mismatch: "+key)
    if abs(left["policy_dt"]-right["policy_dt"]) > 1e-7:
        raise ValueError("Trace control periods differ")
    for key in ("initial_root", "initial_q", "initial_dq", "initial_observation"):
        if not np.allclose(source[key], target[key], rtol=0, atol=1e-7):
            raise ValueError("Unmatched initial state/history: "+key)
    thresholds = {"position_m": position_tolerance, "orientation_rad": orientation_tolerance,
                  "joint_max_rad": joint_tolerance}
    if any(not np.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("Divergence thresholds must be finite and positive")
    count = min(len(source["actions"]), len(target["actions"]))
    if not np.array_equal(source["actions"][:count], target["actions"][:count]):
        raise ValueError("Matched-action comparison requires identical applied actions")
    q1, q2 = source["roots"][:count, 3:7], target["roots"][:count, 3:7]
    q1 = q1/np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2/np.linalg.norm(q2, axis=-1, keepdims=True)
    errors = dict(position_m=np.linalg.norm(source["roots"][:count, :3]-target["roots"][:count, :3], axis=-1),
                  orientation_rad=2*np.arccos(np.clip(np.abs((q1*q2).sum(-1)), 0, 1)),
                  joint_max_rad=np.abs(source["q"][:count]-target["q"][:count]).max(-1),
                  joint_velocity_max_rad_s=np.abs(source["dq"][:count]-target["dq"][:count]).max(-1),
                  observation_rms=np.sqrt(np.mean((source["observations"][:count]-target["observations"][:count])**2, axis=-1)),
                  foot_force_max_n=np.abs(source["forces"][:count]-target["forces"][:count]).max(axis=(1, 2)))
    first = {}
    for name, threshold in thresholds.items():
        hits = np.flatnonzero(errors[name] > threshold)
        first[name] = float(source["time_s"][hits[0]]) if len(hits) else None
    times = [value for value in first.values() if value is not None]
    return dict(schema="x1_transition_comparison_v1", source_engine=left["engine"], target_engine=right["engine"],
                checkpoint_sha256=left["checkpoint_sha256"], compared_steps=count,
                source_steps=len(source["actions"]), target_steps=len(target["actions"]),
                duration_s=float(source["time_s"][count-1]), matched_initial_state_and_actions=True,
                thresholds=thresholds, first_exceeded_s=first,
                first_state_divergence_s=min(times) if times else None,
                maxima={key: float(values.max()) for key, values in errors.items()},
                limits=["Open-loop response diagnostic; not a walking acceptance test.",
                        "Contact forces use each engine's final physical substep; engine sensor semantics and friction-force reporting must be calibrated before comparing force magnitudes.",
                        "Only the common trace prefix is compared; source/target lengths are reported."])
