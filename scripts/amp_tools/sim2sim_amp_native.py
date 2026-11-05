"""Native-phase AMP policy in MuJoCo; controller/observation come from checkpoint."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import torch
from sim2sim_amp import prepare_mjcf,rotation_from_wxyz
from mujoco_diagnostics import foot_contact_forces,roll_pitch_yaw
from native_physics import align_runtime


def policy_frame(command,angular_local,rotation,q,dq,previous,phase,active,default,scales,heading_error=0.,heading_scale=1.,yaw_command=0.):
    # mj_objectVelocity returns angular first, then linear. No measured linear
    # velocity belongs in this actor's98-column input (DH estimates it).
    return np.concatenate(([command*scales['lin_vel'],heading_error*heading_scale,yaw_command*scales['ang_vel']],
        angular_local*scales['ang_vel'],-rotation[2],(q-default)*scales['dof_pos'],
        dq*scales['dof_vel'],previous,
        [math.sin(2*math.pi*phase),math.cos(2*math.pi*phase)] if active else [0.,0.])).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mjcf-source',type=Path,required=True)
    p.add_argument('--meshes',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--video-path',type=Path)
    p.add_argument('--csv-path',type=Path)
    p.add_argument('--training-commit',required=True)
    p.add_argument('--runtime-properties',type=Path,required=True)
    p.add_argument('--passive-probe',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60.)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(5);np.random.seed(5)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    iteration=a.checkpoint.stem.removeprefix('model_')
    video_path=a.video_path or a.checkpoint.parent/('sim2sim_'+iteration+'.mp4')
    csv_path=a.csv_path or a.checkpoint.parent/('sim2sim_'+iteration+'.csv')
    if video_path.exists() or csv_path.exists():raise FileExistsError('Preserve existing recording and CSV')
    video_path.parent.mkdir(parents=True,exist_ok=True)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg,meta=ck['config'],ck['metadata']
    if 'native_reference' not in meta['training']:raise ValueError('Native-cycle checkpoint required')
    names=meta['dof_names'];n=len(names);control=meta['control'];dt=float(meta['policy_dt'])
    simdt=float(cfg['sim']['dt']);decimation=round(dt/simdt)
    assert n==29 and meta['num_obs']==66*98 and abs(decimation*simdt-dt)<1e-7
    sys.path.insert(0,str(a.root))
    from humanoid.amp.dh_models import ActorCriticDHAMP
    from humanoid.amp.motion import MotionLibrary
    from humanoid.amp.native_reference import NativeCycle
    from humanoid.amp.exp20_commands import SmoothCommands
    scaled_phase=meta['training'].get('phase_clock')=='command_scaled_v1'
    if scaled_phase:
        from humanoid.amp.exp29_phase import advance_phase,phase_speed_scale
    policy_cfg=copy.deepcopy(cfg['policy']);exploration=policy_cfg['motion_exploration']
    exploration['indices']=[names.index(k) for k in exploration.pop('joint_names')]
    exploration['command_index']=meta['num_obs']-98
    exploration['command_threshold']=exploration.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    network=ActorCriticDHAMP(meta['num_obs'],cfg['env']['num_privileged_obs'],n,490,98,**policy_cfg)
    network.load_state_dict(ck['model_state_dict'],strict=True);network.eval()
    actor=torch.jit.script(network.get_policy_module().eval());actor.save(str(a.out_dir/'actor.pt'))
    with torch.no_grad():
        probe=torch.randn(4,meta['num_obs'])*.1
        torch.testing.assert_close(actor(probe),network.act_inference(probe),rtol=1e-5,atol=1e-6)
    urdf=Path(cfg['asset']['file'].format(LEGGED_GYM_ROOT_DIR=a.root))
    motion_dir=Path(cfg['amp']['motion_dir'].format(LEGGED_GYM_ROOT_DIR=a.root))
    for stem,sha in meta['training']['reference_source_sha256'].items():
        assert hashlib.sha256((motion_dir/(stem+'.pkl')).read_bytes()).hexdigest()==sha
    lib=MotionLibrary(motion_dir,urdf,names,cfg['amp']['key_body_names'],dt=dt,
        history_steps=cfg['amp']['history_steps'],weights=cfg['amp']['motion_weights'],
        ground_clearance=cfg['amp']['ground_clearance'])
    correction_path=Path(cfg['amp']['native_correction_file'].format(LEGGED_GYM_ROOT_DIR=a.root))
    correction=json.loads(correction_path.read_text())
    canonical=hashlib.sha256(json.dumps(correction,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert canonical==meta['training']['native_reference']['correction_sha256']
    ref=NativeCycle(lib,cfg['amp']['native_clip'],cfg['amp']['native_first_frame'],cfg['amp']['native_last_frame'],contact_correction=correction)
    assert abs(ref.period-meta['training']['native_reference']['period_s'])<1e-7
    xml=prepare_mjcf(a.mjcf_source,a.meshes.resolve().as_posix(),a.out_dir/'mjcf',names)
    m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=simdt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
    d=mujoco.MjData(m)
    joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,k) for k in names]
    assert min(joints)>=0 and m.nu==n
    qa=m.jnt_qposadr[joints];da=m.jnt_dofadr[joints]
    actnames=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0])) for i in range(m.nu)]
    assert sorted(actnames)==sorted(names)
    ai=np.array([actnames.index(k) for k in names])
    # AssetOptions.armature is link-space inertia, not joint-space armature.
    # Preserve the raw URDF/MJCF baseline audit below, then apply runtime data.
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    friction=float(cfg['terrain']['static_friction'])
    m.geom_friction[:,0]=friction
    base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,k) for k in cfg['amp']['key_body_names']]
    assert min(bids)>=0
    default=np.array([cfg['init_state']['default_joint_angles'][k] for k in names])
    standing=np.array(control['standing_state']);limits=np.array(control['hard_limits'])
    assert np.allclose(m.jnt_range[joints],limits,atol=1e-5,rtol=0),'MJCF joint limits differ from checkpoint'
    kp,kd,effort=[np.array(control[k]) for k in ('stiffness','damping','torque_limits')]
    d.qpos[:3]=standing[:3];d.qpos[3:7]=standing[[6,3,4,5]];d.qpos[qa]=standing[13:42]
    mujoco.mj_forward(m,d)
    # Verify all tracked link transforms and all available per-link inertias.
    fk_error=0.
    for q in [standing[13:42],np.clip(default+.15,limits[:,0],limits[:,1]),np.clip(default-.15,limits[:,0],limits[:,1])]:
        d.qpos[qa]=q;mujoco.mj_forward(m,d);r=rotation_from_wxyz(d.qpos[3:7])
        got=(d.xpos[bids]-d.xpos[base])@r
        expected=lib.fk.key_positions(torch.tensor(q[None],dtype=torch.float32))[0].numpy()
        fk_error=max(fk_error,float(np.abs(got-expected).max()))
    assert fk_error<1e-5,('MJCF differs from trained URDF',fk_error)
    compared=0;mass_error=inertia_error=0.
    for link in ET.parse(urdf).getroot().findall('link'):
        inertial=link.find('inertial');bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,link.get('name'))
        if inertial is None or bid<0:continue
        mass=float(inertial.find('mass').get('value'));vals=inertial.find('inertia').attrib
        tensor=np.array([[float(vals['ixx']),float(vals['ixy']),float(vals['ixz'])],
            [float(vals['ixy']),float(vals['iyy']),float(vals['iyz'])],
            [float(vals['ixz']),float(vals['iyz']),float(vals['izz'])]])
        mass_error=max(mass_error,abs(m.body_mass[bid]-mass))
        inertia_error=max(inertia_error,float(np.abs(np.sort(m.body_inertia[bid])-np.linalg.eigvalsh(tensor)).max()));compared+=1
    assert compared>=30 and mass_error<1e-5 and inertia_error<1e-5,(compared,mass_error,inertia_error)
    physics_alignment=align_runtime(m,ck,a.runtime_properties,a.passive_probe,asset_urdf=urdf)
    d.qpos[qa]=standing[13:42];d.qvel[:]=0;mujoco.mj_forward(m,d)
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
    foot_geoms=[[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and m.geom_contype[g]!=0] for bid in feet]
    def soles():
        result=[]
        for geoms in foot_geoms:
            points=[]
            for g in geoms:
                assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
                mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
                points.append(m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            result.append(float(np.concatenate(points)[:,2].min()))
        return result
    z_shift=cfg['amp']['ground_clearance']-min(soles());d.qpos[2]+=z_shift;mujoco.mj_forward(m,d)
    hold=round(cfg['amp']['stand_buffer_seconds']/dt)
    schedule=SmoothCommands(1,'cpu',hold,dt,cfg['commands']['transition_seconds'],cfg['commands']['native_speed'])
    schedule.reset(torch.tensor([0]),torch.tensor([True]));schedule.requested[0,0]=cfg['commands']['native_speed']
    phase=float(ref.nearest_phase(torch.tensor(d.qpos[qa][None],dtype=torch.float32),torch.zeros(1,n))[0])
    previous=np.zeros(n);scales=cfg['normalization']['obs_scales']
    heading_mode=meta.get('training',{}).get('heading_observation')=='vy_command_slot_v1'
    heading_clip=float(meta.get('training',{}).get('heading_clip',1.0))
    heading_obs_scale=float(meta.get('training',{}).get('heading_obs_scale',1.0))
    heading_gain=float(meta.get('training',{}).get('heading_command_gain',0.0))
    heading_cmd_clip=float(meta.get('training',{}).get('heading_command_clip',0.3))
    heading_threshold=float(meta.get('training',{}).get('heading_command_threshold',0.05))
    heading_reference=float(roll_pitch_yaw(rotation_from_wxyz(d.qpos[3:7]))[2]) if heading_mode else 0.
    def heading_error(rotation):
        if not heading_mode:return 0.
        yaw=float(roll_pitch_yaw(rotation)[2])
        return (yaw-heading_reference+math.pi)%(2*math.pi)-math.pi
    def state():
        velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,base,velocity,1)
        return np.array(d.qpos[qa]),np.array(d.qvel[da]),velocity,rotation_from_wxyz(d.qpos[3:7])
    def frame():
        q,dq,velocity,r=state();active=bool(schedule.value[0]>1e-7 or schedule.transition[0])
        err=heading_error(r);obs=float(np.clip(err,-heading_clip,heading_clip))
        wz=float(np.clip(-heading_gain*err,-heading_cmd_clip,heading_cmd_clip)) if active else 0.
        return policy_frame(float(schedule.value[0]),velocity[:3],r,q,dq,previous,phase,active,default,scales,
            obs,heading_obs_scale,wz)
    history=collections.deque([np.zeros(98,dtype=np.float32) for _ in range(65)]+[frame()],maxlen=66)
    renderer=mujoco.Renderer(m,height=720,width=1280)
    video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(1280,720))
    assert video.isOpened(),'MP4 writer unavailable'
    camera=mujoco.MjvCamera();camera.distance=3.0;camera.elevation=-15.;camera.azimuth=90.
    rows=[];fallen=False;fall_time=None;frames=0
    columns=['step','time_s','command','phase','x','y','z','vx','vy','vz','wx','wy','wz','sole_left','sole_right','upright','fallen']
    columns += ['q_'+k for k in names]+['dq_'+k for k in names]+['action_'+k for k in names]+['tau_'+k for k in names]
    columns += ['phase_after','phase_rate_hz','roll','pitch','yaw']
    columns += [side+'_force_'+axis for side in ('left','right') for axis in ('x','y','z')]
    columns += ['left_contact','right_contact']
    columns += ['reference_q_'+k for k in names]+['reference_dq_'+k for k in names]+['pd_target_'+k for k in names]
    try:
        for step in range(round(a.seconds/dt)):
            obs=np.clip(np.concatenate(history),-cfg['normalization']['clip_observations'],cfg['normalization']['clip_observations'])
            with torch.no_grad():action=actor(torch.from_numpy(obs)[None])[0].numpy()
            action=np.clip(action,-cfg['normalization']['clip_actions'],cfg['normalization']['clip_actions'])
            assert np.isfinite(action).all()
            target=np.clip(default+cfg['control']['action_scale']*action,limits[:,0],limits[:,1]);previous=action
            applied=float(schedule.value[0]);active=bool(schedule.value[0]>1e-7 or schedule.transition[0])
            previous_phase=phase
            for _ in range(decimation):
                torque=np.clip(kp*(target-d.qpos[qa])-kd*d.qvel[da],-effort,effort)
                d.ctrl[ai]=torque;mujoco.mj_step(m,d)
            assert np.isfinite(d.qvel).all()
            q,dq,velocity,r=state();height=soles();time=(step+1)*dt
            phase_scale=float(phase_speed_scale(applied,cfg['commands']['native_speed'])) if scaled_phase else 1.
            if active:
                phase=float(advance_phase(phase,applied,dt,ref.period,cfg['commands']['native_speed'])) if scaled_phase else (phase+dt/ref.period)%1.
            blend,blend_derivative=schedule.blend()
            blend_options={'phase_rate_scale':torch.tensor([phase_scale])} if scaled_phase else {}
            ref_state=ref.blended_pose(torch.tensor([phase]),torch.tensor(standing,dtype=torch.float32),blend,blend_derivative,
                **blend_options)[0]
            contact_force,contact_flags=foot_contact_forces(m,d,foot_geoms)
            if (d.qpos[2]<.35 or r[2,2]<.5) and not fallen:fallen=True;fall_time=time
            rows.append([step,time,applied,previous_phase,*d.qpos[:3],*velocity[3:],*velocity[:3],*height,r[2,2],int(fallen),*q,*dq,*action,*torque,
                phase,phase_scale/ref.period if active else 0.,*roll_pitch_yaw(r),*contact_force.flatten(),*contact_flags,
                *ref_state[13:42].tolist(),*ref_state[42:].tolist(),*target])
            if step%2==0:
                camera.lookat[:]=[d.qpos[0],d.qpos[1],max(.4,d.qpos[2])]
                renderer.update_scene(d,camera=camera);pixels=renderer.render()[:,:,::-1].copy()
                text='%s MuJoCo | t=%.2fs | cmd=%.3f | vx=%.3f%s'%(a.checkpoint.parent.name,time,applied,velocity[3],' | FALL' if fallen else '')
                cv2.putText(pixels,text,(18,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,240) if fallen else (20,20,20),2)
                video.write(pixels);frames+=1
            # Same order as Gym: physical frame append, command advance, then
            # synchronize the newest command and phase columns in the history.
            history.append(frame())
            schedule.advance(torch.tensor([False]))
            if bool(schedule.started[0] and schedule.start[0]<=1e-7 and schedule.active_goal[0]>0):
                phase=float(ref.nearest_phase(torch.tensor(q[None],dtype=torch.float32),torch.tensor(dq[None],dtype=torch.float32))[0])
            err=heading_error(r);obs=float(np.clip(err,-heading_clip,heading_clip))
            wz=float(np.clip(-heading_gain*err,-heading_cmd_clip,heading_cmd_clip)) if active else 0.
            history[-1][:3]=[float(schedule.value[0])*scales['lin_vel'],obs*heading_obs_scale,wz*scales['ang_vel']]
            active=bool(schedule.value[0]>1e-7 or schedule.transition[0])
            history[-1][-2:]=[math.sin(2*math.pi*phase),math.cos(2*math.pi*phase)] if active else [0.,0.]
            if step%250==0:print('SIM2SIM step={} time={:.2f} fallen={}'.format(step,time,fallen),flush=True)
            if fallen and time>=fall_time+3.:break
    finally:
        video.release();renderer.close()
    with csv_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
    numeric=np.asarray(rows);settled=numeric[(numeric[:,1]>=cfg['amp']['stand_buffer_seconds']+cfg['commands']['transition_seconds']+2)&(numeric[:,16]==0)]
    summary=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),training_commit=a.training_commit,
        mujoco_version=mujoco.__version__,policy_input=6468,policy_output=29,policy_dt=dt,sim_dt=simdt,decimation=decimation,
        requested_vx=cfg['commands']['native_speed'],phase_rate_hz=1/ref.period,stand_seconds=hold*dt,
        command_transition_seconds=cfg['commands']['transition_seconds'],duration_s=float(numeric[-1,1]),
        video_frames=frames,video_fps=25,fallen=fallen,fall_time_s=fall_time,
        video_path=str(video_path.resolve()),csv_path=str(csv_path.resolve()),
        settled_mean_vx=float(settled[:,7].mean()) if len(settled) else None,
        mjcf_urdf_fk_max_error_m=fk_error,compared_inertial_links=compared,mass_max_error=mass_error,inertia_eigenvalue_max_error=inertia_error,
        initial_ground_shift_m=z_shift,stiffness=kp.tolist(),damping=kd.tolist(),torque_limits=effort.tolist(),
        action_scale=cfg['control']['action_scale'],physics_alignment=physics_alignment,friction=friction,
        gyro_source='mj_objectVelocity XBODY local angular velocity[:3]',reference='checkpoint-native-cycle',
        phase_clock='command_scaled_v1' if scaled_phase else 'native_constant_v1',
        heading_observation='vy_command_slot_v1' if heading_mode else None, heading_clip=heading_clip if heading_mode else None,
        heading_obs_scale=heading_obs_scale if heading_mode else None,
        heading_command_gain=heading_gain if heading_mode else None,
        heading_command_clip=heading_cmd_clip if heading_mode else None,
        heading_command_threshold=heading_threshold if heading_mode else None,
        contact_force_definition='World force on each foot, contact-frame transformed; last physical substep',
        controller='substep bounded PD; no LPF; no noise or pushes',
        termination='base z<0.35 or upright<0.5, record3s after first fall; no resets',
        limits=['MuJoCo convex mesh contact and solver differ from PhysX triangle-mesh/contact-offset handling.',
            'One nominal sim2sim rollout; other initial states and contact conditions are not evaluated.'])
    (a.out_dir/'sim2sim_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
