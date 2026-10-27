"""Audit COM versus lowest-material-point speed using recorded q/dq and root twist."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.motion import URDFKinematics, quat_mul, quat_rotate
from humanoid.amp.grounding import FootGrounding


def audit(model, csv_path, com_offsets):
    torch.set_num_threads(1)
    meta = torch.load(model, map_location="cpu", weights_only=False)["metadata"]
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = meta["dof_names"]
    # One uninterrupted forward window; avoid reset transforms and transients.
    mask = (data["time_s"] >= 23.6) & (data["time_s"] < 29.8)
    if data["done"][mask].any() or not np.allclose(data["command_applied"][mask], .6, atol=1e-6):
        raise ValueError("Require the settled .6 replay window without resets")
    def tensor(keys):
        return torch.tensor(np.column_stack([data[k][mask] for k in keys]), dtype=torch.float64)
    q, dq = tensor(["q_"+n for n in names]), tensor(["dq_"+n for n in names])
    pos, rot = tensor(["x","y","z"]), tensor(["qx","qy","qz","qw"])
    vel = tensor(["world_vx","world_vy","world_vz"])
    omega = torch.cat((tensor(["world_wx","world_wy","world_wz"]), torch.zeros(len(q),1)), -1)
    root_com = torch.tensor(com_offsets['base_link'],dtype=torch.float64).expand_as(vel)
    vel = vel-torch.cross(omega[:,:3],quat_rotate(rot,root_com),dim=-1)
    qdot = .5*quat_mul(omega, rot)
    urdf = ROOT/"resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
    fk = URDFKinematics(urdf, names, [s+"_ankle_roll_link" for s in ("left","right")])
    grounding = FootGrounding(urdf, fk, .002)
    fk.chain = [(a,b,x.double(),r.double(),axis.double(),i) for a,b,x,r,axis,i in fk.chain]
    def transforms(delta):
        root_q = rot+delta*qdot
        root_q /= root_q.norm(dim=-1,keepdim=True)
        local_p, local_q = fk.key_transforms(q+delta*dq)
        root_q = root_q[:,None,:].expand(-1,2,-1)
        return pos[:,None,:]+delta*vel[:,None,:]+quat_rotate(root_q,local_p), quat_mul(root_q,local_q)
    origins, quats = transforms(0.)
    logged = tensor(["foot_x_l","foot_y_l","foot_z_l","foot_x_r","foot_y_r","foot_z_r"]).reshape(-1,2,3)
    error = float((origins-logged).abs().max())
    if error > 1e-4:
        raise ValueError("FK differs from simulator foot positions")
    selected, heights = [], []
    for index, raw_mesh in grounding.feet:
        mesh = raw_mesh.double()
        x,y,z,w = quats[:,index].unbind(-1)
        row = torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1)
        lowest, ids = (row@mesh.T).min(-1)
        heights.append(lowest+origins[:,index,2])
        selected.append(mesh[ids])
    material = torch.stack(selected,1)
    foot_com = torch.tensor([com_offsets[s+'_ankle_roll_link'] for s in ('left','right')],dtype=torch.float64).expand_as(material)
    def velocities(epsilon):
        plus, qp = transforms(epsilon);minus,qm = transforms(-epsilon)
        point_p = plus+quat_rotate(qp,material);point_m = minus+quat_rotate(qm,material)
        com_p = plus+quat_rotate(qp,foot_com);com_m = minus+quat_rotate(qm,foot_com)
        return (com_p-com_m)/(2*epsilon), (point_p-point_m)/(2*epsilon)
    com_v, point_v = velocities(1e-4)
    _, point_check = velocities(2e-4)
    height = torch.stack(heights,-1)
    com_speed, point_speed = com_v[...,:2].norm(dim=-1), point_v[...,:2].norm(dim=-1)
    deficit = ((.015-height)/.015).clamp(0,1).square()
    feet = {}
    for i,side in enumerate(("left","right")):
        contact = tensor(["foot_force_"+s for s in ("l","r")])[:,i] > 20
        near = contact & (height[:,i] < .005)
        pivot = near & (com_speed[:,i] > .05) & (point_speed[:,i] < .02)
        feet[side] = dict(contact_near_floor_samples=int(near.sum()),
                          candidate_rolling_samples=int(pivot.sum()),
                          mean_com_speed_m_s=float(com_speed[:,i][near].mean()),
                          mean_lowest_point_speed_m_s=float(point_speed[:,i][near].mean()),
                          legacy_mean_scuff_cost=float((deficit[:,i]*com_speed[:,i]).mean()),
                          point_mean_scuff_cost=float((deficit[:,i]*point_speed[:,i]).mean()))
    return dict(window_s=[23.6,29.8],samples=len(q),foot_fk_max_error_m=error,
                epsilon_sensitivity_max_m_s=float((point_v-point_check).abs().max()),feet=feet,
                method="COM-aware central directional FK derivative along recorded joint dq and root world twist; same material vertex selected at current minimum z",
                limits="Kinematic estimate, not an observed per-point contact sensor. Lowest vertex may differ from the solver contact patch; this diagnoses reward semantics, not a proven gait cause.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--com-offsets",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.model,args.csv,json.loads(args.com_offsets.read_text()))
    result.update(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
