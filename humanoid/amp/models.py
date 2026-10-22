"""PPO actor/critic and LSGAN motion discriminator."""
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
                 critic_hidden_dims, init_noise_std):
        super().__init__()
        self.actor = mlp(num_obs, actor_hidden_dims, num_actions)
        self.critic = mlp(num_critic_obs, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())

    def distribution(self, obs):
        mean = self.actor(obs)
        return Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean))

    def act_inference(self, obs):
        return self.actor(obs)

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
