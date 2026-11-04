"""Broaden contact anticipation in PhysX; retain nominal controller and observations."""
import json
import torch
from humanoid.amp.contact_domain import sample_offsets,set_contact_offsets
from .x1_amp_exp29_env import X1AMPExp29Env


class X1AMPExp211Env(X1AMPExp29Env):
    def _process_rigid_shape_props(self,props,env_id):
        props=super()._process_rigid_shape_props(props,env_id)
        if env_id==0:
            if self.cfg.domain_rand.randomize_contact_offset:
                self.contact_offset_values=sample_offsets(self.num_envs,self.cfg.domain_rand.contact_offset_range)
            else:
                self.contact_offset_values=torch.full((self.num_envs,),float(self.cfg.sim.physx.contact_offset))
        return set_contact_offsets(props,self.contact_offset_values[env_id])

    def _init_buffers(self):
        super()._init_buffers()
        self.contact_offsets=self.contact_offset_values.to(self.device)
        checked=[]
        for index in sorted({0,self.num_envs//2,3*self.num_envs//4}):
            props=self.gym.get_actor_rigid_shape_properties(self.envs[index],self.actor_handles[index])
            expected=float(self.contact_offset_values[index])
            if not props or any(abs(p.contact_offset-expected)>1e-7 or abs(p.rest_offset)>1e-7 for p in props):
                raise ValueError('Actor did not retain requested contact offset')
            checked.append(dict(env_id=index,contact_offset_m=expected,shapes=len(props)))
        print('CONTACT_RANDOMIZATION_READY '+json.dumps(dict(verified=True,
            assigned_min_m=float(self.contact_offsets.min()),assigned_max_m=float(self.contact_offsets.max()),
            near_fraction=float((self.contact_offsets<=.00100001).float().mean()),
            original_fraction=float((self.contact_offsets==.01).float().mean()),
            sampled_actor_checks=checked)),flush=True)

    def compute_reward(self):
        super().compute_reward()
        self.extras['training_metrics']['Contact/offset_m']=self.contact_offsets

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        result.update(curriculum='contact_margin_robustness_v1',
            robustness_scope='contact_offset coverage plus unchanged source friction; original PD and rigid-body properties',
            contact_offset_randomization=dict(range_m=[.001,.020],near_fraction=.5,original_fraction=.25,
                random_fraction=.25,original_m=.01,rest_offset_m=0.,resample='static_per_environment'))
        return result
