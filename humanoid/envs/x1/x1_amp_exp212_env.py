"""Reserve 25 percent of training environments for deterministic forward startup."""
import json
import torch
from humanoid.amp.nominal_startup import cohort_mask,training_metadata
from .x1_amp_exp211_env import X1AMPExp211Env


class X1AMPExp212Env(X1AMPExp211Env):
    def _nominal(self,env_ids):
        return cohort_mask(env_ids,self.num_envs,self.cfg.amp.nominal_startup_fraction)

    def _process_rigid_shape_props(self,props,env_id):
        props=super()._process_rigid_shape_props(props,env_id)
        if env_id<int(self.num_envs*self.cfg.amp.nominal_startup_fraction):
            for p in props:p.friction=.8
            self.friction_coeffs[env_id]=.8
            self.env_frictions[env_id]=.8
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.nominal_startup_mask=self._nominal(torch.arange(self.num_envs,device=self.device))
        count=int(self.nominal_startup_mask.sum())
        checks=[]
        for index in sorted({0,count-1}):
            props=self.gym.get_actor_rigid_shape_properties(self.envs[index],self.actor_handles[index])
            if not props or any(abs(p.friction-.8)>1e-6 or abs(p.contact_offset-.001)>1e-7 or abs(p.rest_offset)>1e-7 for p in props):
                raise ValueError('Nominal cohort shape properties did not persist')
            checks.append(dict(env_id=index,shapes=len(props),friction=float(props[0].friction),
                contact_offset_m=float(props[0].contact_offset)))
        print('NOMINAL_STARTUP_READY '+json.dumps(dict(verified=True,nominal_envs=count,
            total_envs=self.num_envs,fraction=count/self.num_envs,sampled_actor_checks=checks)),flush=True)

    def _adjust_reset_states(self,env_ids,standing,phase,state):
        standing,phase,state=super()._adjust_reset_states(env_ids,standing,phase,state)
        if self.evaluation_reset_mode is None and not hasattr(self,'diagnostic_reset_phases'):
            nominal=self._nominal(env_ids)
            standing[nominal]=True
            state[nominal]=self.standing_state
            if not getattr(self,'_nominal_reset_verified',False) and nominal.any():
                assert torch.equal(state[nominal],self.standing_state.expand(int(nominal.sum()),-1))
                self._nominal_reset_verified=True
                print('NOMINAL_STARTUP_RESET_VERIFIED '+json.dumps(dict(verified=True,
                    exact_standing_state=True,reset_count=int(nominal.sum()),joint_noise=0.)),flush=True)
        return standing,phase,state

    def _resample_commands(self,env_ids):
        nominal=self._nominal(env_ids)
        if self._reset_commands is not None:
            self._reset_commands[nominal]=self.commands.new_tensor([self.cfg.commands.native_speed,0.,0.])
        super()._resample_commands(env_ids)
        self.standing_commands.requested[env_ids[nominal]]=self.commands.new_tensor([self.cfg.commands.native_speed,0.,0.])

    def compute_observations(self):
        super().compute_observations()
        mask=self._nominal(torch.arange(self.num_envs,device=self.device))
        frame=self._policy_frame()
        self.obs_history[-1][mask]=frame[mask]
        self.obs_buf[mask,-frame.shape[-1]:]=frame[mask]
        if not getattr(self,'_nominal_obs_verified',False) and mask.any():
            assert torch.equal(self.obs_buf[mask,-frame.shape[-1]:],frame[mask])
            self._nominal_obs_verified=True
            print('NOMINAL_STARTUP_OBS_VERIFIED '+json.dumps(dict(verified=True,
                nominal_envs=int(mask.sum()),frame_dimension=frame.shape[-1],noise_max_error=0.)),flush=True)

    def compute_reward(self):
        super().compute_reward()
        mask=self.nominal_startup_mask
        moving=self.standing_commands.value>1e-7
        done=self.reset_buf.bool()&mask
        metrics=self.extras['training_metrics']
        metrics.update({'Nominal/fraction':mask.float(),
            'Nominal/episode_seconds':(self.episode_length_buf*self.dt)[done],
            'Nominal/moving_abs_yaw_rate':self.base_ang_vel[mask&moving,2].abs(),
            'Nominal/forward_vx':self.base_lin_vel[mask&moving,0],
            'Nominal/non_timeout_rate':(~self.time_out_buf[done]).float()})

    def checkpoint_training_metadata(self):
        return training_metadata(super().checkpoint_training_metadata())
