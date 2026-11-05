"""One-environment, read-only PhysX/AMP audit; never creates a PPO runner."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def vec(value):
    return [float(getattr(value, axis)) for axis in 'xyz']


def bodies(properties):
    # Mat33 x/y/z vectors are columns; all inertia tensors are symmetric.
    return [dict(mass=float(p.mass), com=vec(p.com),
                 inertia=[[float(getattr(getattr(p.inertia, c), r)) for c in 'xyz'] for r in 'xyz'])
            for p in properties]


def apply_config(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(target, key) and not isinstance(getattr(target, key), dict):
            apply_config(getattr(target, key), value)
        else:
            setattr(target, key, copy.deepcopy(value))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=750)
    parser.add_argument('--passive-probe-only', action='store_true')
    audit, remaining = parser.parse_known_args()
    audit.output_dir.mkdir(parents=True, exist_ok=True)
    if (audit.output_dir/'runtime.json').exists():
        raise FileExistsError('Preserve earlier runtime export')
    sys.path.insert(0, str(audit.root.resolve()))
    sys.argv = [sys.argv[0]] + remaining
    import isaacgym
    import numpy as np
    import torch
    from humanoid.envs import task_registry
    from humanoid.utils import get_args
    from humanoid.envs.base.legged_robot import LeggedRobot
    from humanoid.envs.x1.x1_amp_env import X1AMPEnv
    from humanoid.amp.dh_models import ActorCriticDHAMP
    from humanoid.amp.models import Discriminator
    torch.set_num_threads(1)
    ck = torch.load(audit.checkpoint, map_location='cpu')
    cfg, _ = task_registry.get_cfgs('x1_amp_exp29')
    apply_config(cfg, ck['config'])
    cfg.seed = 5
    cfg.env.num_envs = 1
    cfg.env.enable_camera_sensors = False
    cfg.env.capture_replay_state = True
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.friction_range = [float(cfg.terrain.static_friction)]*2
    cfg.domain_rand.restitution_range = [0., 0.]
    cfg.commands.guidance_evaluation = True
    cfg.commands.ranges.lin_vel_x = [cfg.commands.native_speed]*2
    cfg.amp.rsi_probability = 0.
    args = get_args()
    if args.num_envs != 1 or args.seed != 5 or not args.headless:
        raise ValueError('Requires --num_envs=1 --seed=5 --headless')
    imported = []
    old_process = LeggedRobot._process_rigid_body_props

    def capture_process(self, props, env_id):
        if env_id == 0:
            imported.extend(bodies(props))
        return old_process(self, props, env_id)

    LeggedRobot._process_rigid_body_props = capture_process
    old_reward = X1AMPEnv.compute_reward

    def capture_reward(self):
        # Capture the same physical instant used by AMP, before automatic reset.
        key = self.rigid_state[:, self.key_body_indices, :3].clone()
        old_reward(self)
        self.extras['audit_key_world'] = key

    X1AMPEnv.compute_reward = capture_reward
    env = None
    try:
        env, _ = task_registry.make_env('x1_amp_exp29', args=args, env_cfg=cfg)
        env.evaluation_reset_mode = 'standing'
        curriculum = ck.get('infos', {}).get('exp21', {}).get('learning_state', {}).get('curriculum')
        if curriculum is None:
            raise ValueError('Missing checkpoint curriculum; do not guess AMP stage')
        env.gait_curriculum.load_state_dict(curriculum)
        env.set_training_iteration(ck['iter'])
        gym, handle, actor = env.gym, env.envs[0], env.actor_handles[0]
        body_names = gym.get_actor_rigid_body_names(handle, actor)
        final = bodies(gym.get_actor_rigid_body_properties(handle, actor))
        for seq in (imported, final):
            for name, item in zip(body_names, seq):
                item['name'] = name
        dof = gym.get_actor_dof_properties(handle, actor)
        shapes = []
        for prop in gym.get_actor_rigid_shape_properties(handle, actor):
            shapes.append({k: float(getattr(prop, k)) for k in
                           ('friction', 'rolling_friction', 'torsion_friction', 'restitution',
                            'compliance', 'thickness', 'contact_offset', 'rest_offset') if hasattr(prop, k)})
        runtime = dict(checkpoint_sha256=hashlib.sha256(audit.checkpoint.read_bytes()).hexdigest(),
            asset_urdf_sha256=hashlib.sha256(Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=audit.root)).read_bytes()).hexdigest(),
            iteration=ck['iter'], seed=5, num_envs=1, training_updates=0,
            source='Exact checkpoint configuration and archived production environment; nominal friction0.8',
            before_recompute_inertia=imported, final_bodies=final,
            dofs={name: {k: np.asarray(dof[k][i]).item() for k in dof.dtype.names}
                  for i, name in enumerate(env.dof_names)}, shapes=shapes,
            stiffness=env.p_gains.cpu().tolist(), damping=env.d_gains.cpu().tolist(),
            torque_limits=env.torque_limits.cpu().tolist(), env_origin=env.env_origins[0].cpu().tolist(),
            policy_dt=float(env.dt), sim_dt=float(env.sim_params.dt),
            asset_config=ck['config']['asset'], sim_config=ck['config']['sim'],
            self_collision_filter=cfg.asset.self_collisions)
        (audit.output_dir/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
        if audit.passive_probe_only:
            from isaacgym import gymtorch
            # Suspended free body, no PD/efforts/contact: isolate whether the
            # imported drive damping acts in the production EFFORT mode.
            results = []
            for damping in (1., 0., 1.):
                props = gym.get_actor_dof_properties(handle, actor)
                props['damping'][:] = damping
                gym.set_actor_dof_properties(handle, actor, props)
                env.root_states[:] = torch.tensor(ck['metadata']['control']['standing_state'][:13], device=env.device)
                env.root_states[:, :3] = env.env_origins + env.root_states.new_tensor([0., 0., 3.])
                env.root_states[:, 7:13] = 0.
                env.dof_pos[:] = torch.tensor(ck['metadata']['control']['standing_state'][13:42], device=env.device)
                env.dof_vel[:] = torch.linspace(-.3, .3, 29, device=env.device)
                gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
                start = torch.cat((env.root_states[0], env.dof_pos[0], env.dof_vel[0])).cpu().tolist()
                for _ in range(10):
                    gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(torch.zeros_like(env.torques)))
                    gym.simulate(env.sim)
                    gym.fetch_results(env.sim, True)
                gym.refresh_dof_state_tensor(env.sim)
                gym.refresh_actor_root_state_tensor(env.sim)
                gym.refresh_net_contact_force_tensor(env.sim)
                results.append(dict(drive_damping=damping, start=start,
                    end=torch.cat((env.root_states[0], env.dof_pos[0], env.dof_vel[0])).cpu().tolist(),
                    maximum_contact_force=float(env.contact_forces.abs().max())))
            velocity_delta = float(np.max(np.abs(np.asarray(results[0]['end'])[42:]-np.asarray(results[1]['end'])[42:])))
            repeat_delta = float(np.max(np.abs(np.asarray(results[0]['end'])-np.asarray(results[2]['end']))))
            report = dict(results=results, damping_effect_max_joint_velocity_delta=velocity_delta,
                          repeat_max_state_delta=repeat_delta, dt=float(env.sim_params.dt), steps_per_case=10,
                          training_updates=0)
            (audit.output_dir/'passive_probe.json').write_text(json.dumps(report, indent=2)+'\n')
            print('PASSIVE_PROBE '+json.dumps({k:v for k,v in report.items() if k!='results'}), flush=True)
            return
        policy_cfg = copy.deepcopy(ck['config']['policy'])
        names = env.dof_names
        exploration = policy_cfg['motion_exploration']
        exploration['indices'] = [names.index(n) for n in exploration.pop('joint_names')]
        exploration['command_index'] = ck['metadata']['num_obs']-98
        exploration['command_threshold'] = exploration.pop('speed_threshold')*cfg.normalization.obs_scales.lin_vel
        net = ActorCriticDHAMP(ck['metadata']['num_obs'], cfg.env.num_privileged_obs, 29, 490, 98,
                              **policy_cfg).to(env.device)
        net.load_state_dict(ck['model_state_dict'], strict=True)
        net.eval()
        disc = Discriminator(env.motion_library.feature_dim, cfg.amp.history_steps,
                             ck['config']['discriminator']['hidden_dims']).to(env.device)
        disc.load_state_dict(ck['discriminator_state_dict'], strict=True)
        disc.eval()
        obs, _ = env.reset()
        env.set_evaluation_command(cfg.commands.native_speed)
        samples = {k: [] for k in ('state', 'key_world', 'amp', 'disc_score', 'style_score',
            'task_reward', 'multiplier', 'valid', 'moving', 'transition', 'done', 'timeout',
            'phase', 'command', 'hold', 'velocity', 'omega', 'gravity', 'observation', 'action')}
        for step in range(audit.steps):
            if step == 500:
                obs = env.set_evaluation_command(0.)
            with torch.no_grad():
                action = net.act_inference(obs)
                obs, _, reward, done, info = env.step(action)
                record = info['replay_state']
                state = torch.cat((record['root'][0], record['q'][0], record['dq'][0]))
                state = state.clone()
                state[:3] -= env.env_origins[0]
                values = dict(state=state, key_world=info['audit_key_world'][0]-env.env_origins[0],
                    amp=info['amp_obs'][0], disc_score=disc(info['amp_obs'])[0],
                    style_score=disc.style_reward(info['amp_obs'])[0], task_reward=reward[0],
                    multiplier=info['amp_style_multiplier'][0], valid=info['amp_valid'][0],
                    moving=info['amp_moving'][0], transition=record['command_transition'][0],
                    done=done[0], timeout=record['timeout'][0], phase=record['reference_phase'][0],
                    command=record['command'][0], hold=record['stand_steps_left'][0],
                    velocity=record['velocity'][0], omega=record['omega'][0],
                    gravity=record['gravity'][0], observation=obs[0], action=action[0])
                for key, value in values.items():
                    samples[key].append(value.detach().cpu().numpy().copy())
            if step % 250 == 0:
                print('AUDIT_SAMPLE {}'.format(step), flush=True)
        np.savez_compressed(str(audit.output_dir/'samples.npz'), **{k: np.stack(v) for k, v in samples.items()})
        print('AUDIT_COMPLETE steps={} training_updates=0'.format(audit.steps), flush=True)
    finally:
        LeggedRobot._process_rigid_body_props = old_process
        X1AMPEnv.compute_reward = old_reward
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
