"""Original DH policy topology adapted to the 29-DOF AMP runner.

As in ActorCriticDH, history frames are Conv1d channels and convolution runs
along the per-frame feature axis. The actor receives short history, estimated
scaled body velocity, and the 64-dimensional long-history embedding.
"""
import torch
from torch import nn

from .models import ActorCritic, mlp


class DHPolicy(nn.Module):
    """Exportable policy including both history branches and velocity estimation."""
    def __init__(self, actor, long_history, state_estimator, num_short_obs,
                 in_channels, num_proprio_obs):
        super().__init__()
        self.actor = actor
        self.long_history = long_history
        self.state_estimator = state_estimator
        self.num_short_obs = num_short_obs
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs

    def forward(self, observations):
        short_history = observations[..., -self.num_short_obs:]
        velocity = self.state_estimator(short_history)
        encoded = self.long_history(observations.reshape(-1, self.in_channels, self.num_proprio_obs))
        return self.actor(torch.cat((short_history, velocity, encoded), dim=-1))


class ActorCriticDHAMP(ActorCritic):
    def __init__(self, num_obs, num_critic_obs, num_actions, num_short_obs,
                 num_proprio_obs, in_channels, state_estimator_hidden_dims=(256, 128, 64),
                 kernel_size=(6, 4), filter_size=(32, 16), stride_size=(3, 2),
                 lh_output_dim=64, **policy_cfg):
        if (num_obs != in_channels*num_proprio_obs or num_short_obs <= 0
                or num_short_obs > num_obs or num_short_obs % num_proprio_obs):
            raise ValueError("DH observation dimensions do not match long/short frame stacks")
        if not (len(kernel_size) == len(filter_size) == len(stride_size) and kernel_size):
            raise ValueError("DH CNN kernel/filter/stride lists must have the same non-zero length")
        if lh_output_dim <= 0:
            raise ValueError("DH history embedding must have positive size")
        super().__init__(num_obs, num_critic_obs, num_actions,
                         actor_input_dim=num_short_obs+3+lh_output_dim, **policy_cfg)
        self.num_short_obs = num_short_obs
        self.num_proprio_obs = num_proprio_obs
        self.in_channels = in_channels
        self.lh_output_dim = lh_output_dim
        layers, channels, width = [], in_channels, num_proprio_obs
        for filters, kernel, stride in zip(filter_size, kernel_size, stride_size):
            if min(filters, kernel, stride) <= 0 or width < kernel:
                raise ValueError("Invalid DH CNN dimensions")
            layers.extend((nn.Conv1d(channels, filters, kernel, stride), nn.ReLU()))
            width = (width-kernel)//stride+1
            channels = filters
        layers.extend((nn.Flatten(), nn.Linear(width*channels, 128), nn.ELU(),
                       nn.Linear(128, lh_output_dim)))
        self.long_history = nn.Sequential(*layers)
        self.state_estimator = mlp(num_short_obs, state_estimator_hidden_dims, 3)

    def estimate_velocity(self, observations):
        return self.state_estimator(observations[..., -self.num_short_obs:])

    def act_inference(self, observations):
        short_history = observations[..., -self.num_short_obs:]
        velocity = self.state_estimator(short_history)
        encoded = self.long_history(observations.reshape(-1, self.in_channels, self.num_proprio_obs))
        return self.actor(torch.cat((short_history, velocity, encoded), dim=-1))

    def get_policy_module(self):
        return DHPolicy(self.actor, self.long_history, self.state_estimator,
                        self.num_short_obs, self.in_channels, self.num_proprio_obs)
