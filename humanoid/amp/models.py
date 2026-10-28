"""PPO actor/critic and LSGAN motion discriminator."""
import math
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, hidden, outputs):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(inputs, size), nn.ELU()))
        inputs = size
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_critic_obs, num_actions, actor_hidden_dims,
                 critic_hidden_dims, init_noise_std, motion_exploration=None, actor_input_dim=None):
        super().__init__()
        self.actor = mlp(num_obs if actor_input_dim is None else actor_input_dim, actor_hidden_dims, num_actions)
        self.critic = mlp(num_critic_obs, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())
        self.motion_std_floor = 0.
        self.motion_command_index = 0
        self.motion_command_threshold = 0.
        mask = torch.zeros(num_actions, dtype=torch.bool)
        if motion_exploration is not None:
            indices = motion_exploration["indices"]
            floor = float(motion_exploration["std_minimum"])
            index = int(motion_exploration["command_index"])
            threshold = float(motion_exploration["command_threshold"])
            if not indices or len(set(indices)) != len(indices) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= num_actions for i in indices):
                raise ValueError("Invalid motion exploration joints")
            if not math.isfinite(floor) or not 0 < floor <= math.exp(2) or not 0 <= index < num_obs or not math.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid motion exploration settings")
            mask[indices] = True
            self.motion_std_floor, self.motion_command_index, self.motion_command_threshold = floor, index, threshold
        # Device-aware mask without adding keys to existing checkpoint weights.
        self.register_buffer("motion_std_mask", mask, persistent=False)

    def distribution(self, obs):
        mean = self.act_inference(obs)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        if self.motion_std_floor > 0:
            moving = obs[..., self.motion_command_index].abs().gt(self.motion_command_threshold).unsqueeze(-1)
            std = torch.where(moving & self.motion_std_mask, std.clamp(min=self.motion_std_floor), std)
        return Normal(mean, std)

    def act_inference(self, obs):
        return self.actor(obs)

    def get_policy_module(self):
        """The complete deterministic inference graph, without training heads."""
        return self.actor

    def evaluate(self, obs):
        return self.critic(obs).squeeze(-1)


class RunningNormalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        count = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + count
        variance = (self.var*self.count + x.var(0, unbiased=False)*count
                    + delta.square()*self.count*count/total) / total
        self.mean.add_(delta * count/total)
        self.var.copy_(variance)
        self.count.copy_(total)

    def forward(self, x):
        return ((x-self.mean) / (self.var+1e-5).sqrt()).clamp(-10, 10)


class Discriminator(nn.Module):
    def __init__(self, feature_dim, history_steps, hidden_dims):
        super().__init__()
        self.normalizer = RunningNormalizer(feature_dim)
        self.network = mlp(feature_dim*history_steps, hidden_dims, 1)

    def forward(self, history):
        return self.network(self.normalizer(history).flatten(1)).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, history):
        return (1 - 0.25*(self(history)-1).square()).clamp(min=0)


class ReplayBuffer:
    def __init__(self, capacity, shape, device):
        self.data = torch.empty(capacity, *shape, device=device)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    @torch.no_grad()
    def insert(self, x):
        x = x[-self.capacity:]
        n = len(x)
        first = min(n, self.capacity-self.cursor)
        self.data[self.cursor:self.cursor+first].copy_(x[:first])
        self.data[:n-first].copy_(x[first:])
        self.cursor = (self.cursor+n) % self.capacity
        self.size = min(self.capacity, self.size+n)

    def sample(self, n):
        if not self.size:
            raise ValueError("Cannot sample empty AMP replay buffer")
        return self.data[torch.randint(self.size, (n,), device=self.data.device)]
