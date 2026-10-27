"""Empirically identify rigid-state velocity reference using an offset-COM free body."""
from pathlib import Path
import json
import isaacgym
from isaacgym import gymapi, gymtorch
import torch


def main():
    root=Path(__file__).resolve().parent
    (root/'velocity_probe.urdf').write_text('''<robot name="probe"><link name="base">
<inertial><origin xyz="0.2 0 0"/><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link></robot>''')
    gym=gymapi.acquire_gym();params=gymapi.SimParams();params.dt=.001
    params.gravity=gymapi.Vec3(0,0,0);params.up_axis=gymapi.UP_AXIS_Z
    params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=1
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
    options=gymapi.AssetOptions();options.disable_gravity=True;options.override_com=False
    asset=gym.load_asset(sim,str(root),'velocity_probe.urdf',options)
    env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,3),1)
    pose=gymapi.Transform();pose.p.z=2
    actor=gym.create_actor(env,asset,pose,'probe',0,0)
    com=gym.get_actor_rigid_body_properties(env,actor)[0].com
    offset=torch.tensor([com.x,com.y,com.z]);assert offset.norm()>.1
    gym.prepare_sim(sim)
    root_state=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    body=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim);gym.refresh_rigid_body_state_tensor(sim)
    root_state[:,7:13]=0;root_state[:,12]=1
    gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root_state))
    def com_position(state):
        q=state[3:7];v=offset;t=2*torch.cross(q[:3],v,dim=0)
        return state[:3]+v+q[3]*t+torch.cross(q[:3],t,dim=0)
    previous=None
    for _ in range(100):
        previous=body[0].clone()
        gym.simulate(sim);gym.fetch_results(sim,True)
        gym.refresh_actor_root_state_tensor(sim);gym.refresh_rigid_body_state_tensor(sim)
    current=body[0].clone()
    origin_fd=(current[:3]-previous[:3])/params.dt
    com_fd=(com_position(current)-com_position(previous))/params.dt
    reported=current[7:10]
    errors=dict(origin=float((origin_fd-reported).norm()),com=float((com_fd-reported).norm()))
    reference='center_of_mass' if errors['com']<.005 and errors['origin']>.1 else ('body_origin' if errors['origin']<.005 and errors['com']>.1 else 'ambiguous')
    result=dict(reference=reference,com_local=offset.tolist(),body_origin_velocity_fd=origin_fd.tolist(),
                center_of_mass_velocity_fd=com_fd.tolist(),reported_rigid_linear_velocity=reported.tolist(),
                reported_root_linear_velocity=root_state[0,7:10].tolist(),errors=errors,
                method='CPU PhysX, zero gravity, asymmetric COM at0.2m, angular velocity1rad/s,100x0.001s')
    (root/'velocity_reference_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VELOCITY_REFERENCE '+json.dumps(result),flush=True)
    gym.destroy_sim(sim)
    assert reference!='ambiguous'


if __name__=='__main__':main()
