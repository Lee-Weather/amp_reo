#!/usr/bin/env python3
"""Check a standing pose on the URDF with pure forward kinematics.

No Isaac Gym or torch required: parses the URDF with lxml, reads the foot
meshes to recover the sole outline, and reports the quantities that decide
whether a zero-command standing pose is usable:

  * each foot's lowest point and the height gap between the feet
  * the front/back offset between the left and right foot
  * the CoM projection and its margin inside the support polygon
  * joint-limit violations of the pose itself

Example:

  python scripts/amp_tools/check_pose_fk.py --pose pd_stand --out /tmp/pose.json
  python scripts/amp_tools/check_pose_fk.py --pose pd_stand --flip-right-hip-pitch

The pose presets live in this file (``POSE_PRESETS``) because importing the
env configs would require Isaac Gym.  ``--set joint=value`` overrides any
joint without editing the file.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"

DOF_NAMES = (["lumbar_" + x + "_joint" for x in ("yaw", "roll", "pitch")]
             + [side + "_" + x + "_joint" for side in ("left", "right")
                for x in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
                          "elbow_yaw", "wrist_pitch", "wrist_roll")]
             + [side + "_" + x + "_joint" for side in ("left", "right")
                for x in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")])

# exp3.3 deployment pd_stand (czy/origin_x1/agibot_x1_train-main/.../x1_dh_stand_config.py:129-140)
PD_STAND = [
    0.00, 0.00, 0.00,
    0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
    0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
    0.40, 0.05, -0.31, 0.49, -0.21, 0.00,
    -0.40, -0.05, 0.31, 0.49, -0.21, 0.00,
]

# Mean of the four normalized AMP clips, in DOF_NAMES order (x1_amp_config.py:82-87)
MEAN29 = [
    -0.0114, -0.0210, 0.3162,
    -0.0494, -0.1600, -0.0988, 0.5442, 0.1529, 0.2649, 0.2223,
    -0.0808, -0.2659, -0.2642, 0.6477, -0.0512, -0.1349, -0.0339,
    -0.0441, 0.0610, 0.0198, 0.2917, -0.1627, -0.0514,
    -0.0034, -0.0291, 0.0022, 0.3494, -0.1630, 0.0985,
]

POSE_PRESETS = {"pd_stand": PD_STAND, "mean29": MEAN29}
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
SOLE_BAND_M = 0.02


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = xyz
    return T


def parse_urdf(path):
    tree = etree.parse(str(path))
    joints = {}
    links = {}
    for joint in tree.getroot().findall("joint"):
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        axis_el = joint.find("axis")
        axis = [float(v) for v in (axis_el.get("xyz", "1 0 0") if axis_el is not None else "1 0 0").split()]
        limit = joint.find("limit")
        joints[joint.get("name")] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "T": transform(xyz, rpy),
            "axis": axis,
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
        }
    for link in tree.getroot().findall("link"):
        inertial = link.find("inertial")
        mesh = None
        for tag in ("visual", "collision"):
            node = link.find(tag)
            if node is None:
                continue
            geo = node.find("geometry")
            candidate = geo.find("mesh") if geo is not None else None
            if candidate is not None:
                mesh = candidate.get("filename")
                break
        links[link.get("name")] = {
            "mass": float(inertial.find("mass").get("value")) if inertial is not None else 0.0,
            "com": [float(v) for v in inertial.find("origin").get("xyz").split()] if inertial is not None else [0, 0, 0],
            "mesh": mesh,
        }
    return joints, links


def chain_order(joints, root="base_link"):
    children = {}
    for name, joint in joints.items():
        children.setdefault(joint["parent"], []).append(name)
    order = []

    def walk(link):
        for name in children.get(link, []):
            order.append(name)
            walk(joints[name]["child"])

    walk(root)
    return order


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    body = np.frombuffer(data[84:84 + count * 50], dtype=np.uint8).reshape(count, 50)
    verts = body[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return verts.reshape(-1, 3).astype(np.float64)


def hull_2d(points):
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def hull_margin(poly, point):
    """Signed distance from ``point`` to the hull boundary (positive inside)."""
    n = len(poly)
    if n < 3:
        return float("nan")
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    ccw = area > 0
    best = float("inf")
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        inward = np.array([-edge[1], edge[0]]) / norm if ccw else np.array([edge[1], -edge[0]]) / norm
        best = min(best, float(np.dot(point - a, inward)))
    return best


def evaluate(urdf, pose):
    joints, links = parse_urdf(urdf)
    order = chain_order(joints)
    world = {"base_link": np.eye(4)}
    for name in order:
        joint = joints[name]
        angle = pose.get(name, 0.0)
        T = world[joint["parent"]] @ joint["T"] @ \
            np.block([[axis_angle(joint["axis"], angle), np.zeros((3, 1))], [np.zeros((3,)), 1]])
        world[joint["child"]] = T

    urdf_dir = Path(urdf).resolve().parent
    report = {"urdf": str(urdf), "pose": {k: round(float(v), 6) for k, v in pose.items()},
              "feet": {}, "com_world": None, "support": {}, "limits": []}

    sole_world = {}
    for name in FOOT_LINKS:
        mesh = links[name]["mesh"]
        verts = read_stl(urdf_dir / mesh)                         # shape (N, 3) in link frame
        T = world[name]
        w = (T[:3, :3] @ verts.T).T + T[:3, 3]                  # all vertices in world
        low_w = w[:, 2].min()
        band = w[:, 2] <= low_w + SOLE_BAND_M
        sole_world[name] = w[band]
        report["feet"][name] = {
            "sole_points": int(band.sum()),
            "min_z_mm": round(float(low_w) * 1000, 2),
            "sole_x_mm": [round(float(w[band, 0].min()) * 1000, 2),
                          round(float(w[band, 0].max()) * 1000, 2)],
            "centroid_mm": [round(float(v) * 1000, 2) for v in w[band].mean(axis=0)],
        }
    report["base_height_for_contact_m"] = round(
        float(-min(report["feet"][n]["min_z_mm"] for n in FOOT_LINKS)) / 1000, 4)

    masses = {name: links[name]["mass"] for name in links}
    total_mass = sum(masses.values())
    com = np.zeros(3)
    for name, info in links.items():
        if info["mass"] <= 0:
            continue
        T = world[name]
        com += info["mass"] * (T[:3, :3] @ np.asarray(info["com"]) + T[:3, 3])
    com /= total_mass
    report["com_world"] = [round(float(v) * 1000, 2) for v in com]
    report["total_mass_kg"] = round(float(total_mass), 3)

    all_pts = np.concatenate(list(sole_world.values()))
    hull = hull_2d(all_pts[:, :2])
    margin = hull_margin(hull, com[:2])
    report["support"] = {
        "hull_mm": [[round(float(x) * 1000, 1), round(float(y) * 1000, 1)] for x, y in hull],
        "com_margin_mm": round(margin * 1000, 2),
        "com_inside": bool(margin > 0),
    }

    lz = report["feet"]["left_ankle_roll_link"]["min_z_mm"]
    rz = report["feet"]["right_ankle_roll_link"]["min_z_mm"]
    report["checks"] = {
        "foot_height_gap_mm": round(abs(lz - rz), 2),
        "foot_front_gap_mm": round(abs(report["feet"]["left_ankle_roll_link"]["centroid_mm"][0]
                                       - report["feet"]["right_ankle_roll_link"]["centroid_mm"][0]), 2),
        "foot_lateral_gap_mm": round(abs(report["feet"]["left_ankle_roll_link"]["centroid_mm"][1]
                                         - report["feet"]["right_ankle_roll_link"]["centroid_mm"][1]), 2),
        "lowest_point_mm": round(min(lz, rz), 2),
        "com_margin_mm": round(margin * 1000, 2),
    }

    for name, angle in pose.items():
        joint = joints.get(name)
        if joint is None or joint["lower"] is None:
            continue
        if angle < joint["lower"] - 1e-9 or angle > joint["upper"] + 1e-9:
            report["limits"].append({"joint": name, "angle": angle,
                                     "limit": [joint["lower"], joint["upper"]]})
    return report


def main():
    parser = argparse.ArgumentParser(description="Forward-kinematics check of a standing pose")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pose", choices=sorted(POSE_PRESETS), default="pd_stand")
    parser.add_argument("--flip-right-hip-pitch", action="store_true",
                        help="Compare the mirrored convention: right hip pitch +0.4 instead of -0.4")
    parser.add_argument("--set", action="append", default=[], metavar="JOINT=VALUE",
                        help="Override one joint angle in radians, e.g. right_hip_pitch_joint=0.4")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true",
                        help="Print a one-line summary instead of the full JSON report")
    args = parser.parse_args()

    values = list(POSE_PRESETS[args.pose])
    pose = dict(zip(DOF_NAMES, values))
    if args.flip_right_hip_pitch:
        pose["right_hip_pitch_joint"] = -pose["right_hip_pitch_joint"]
    for item in args.set:
        name, _, value = item.partition("=")
        pose[name] = float(value)

    report = evaluate(args.urdf, pose)
    checks = report["checks"]
    verdict = (checks["foot_height_gap_mm"] <= 5.0 and checks["foot_front_gap_mm"] <= 10.0
               and checks["com_margin_mm"] >= 20.0)
    report["verdict"] = "pass" if verdict else "fail"
    if args.quiet:
        feet = report["feet"]
        left, right = feet["left_ankle_roll_link"], feet["right_ankle_roll_link"]
        hull_x = [min(p[0] for p in report["support"]["hull_mm"]),
                  max(p[0] for p in report["support"]["hull_mm"])]
        print(f"{args.pose}{' +flip' if args.flip_right_hip_pitch else ''}: "
              f"dz={checks['foot_height_gap_mm']:.1f} front={checks['foot_front_gap_mm']:.1f} "
              f"lateral={checks['foot_lateral_gap_mm']:.1f} | com_margin={checks['com_margin_mm']:.1f}mm "
              f"com_x={report['com_world'][0]:.0f} foot_x={(left['centroid_mm'][0] + right['centroid_mm'][0]) / 2:.0f} "
              f"hull_x=[{hull_x[0]:.0f},{hull_x[1]:.0f}] | {report['verdict']}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()