"""Reuse bounded control and resets, tighten only linear velocity tracking."""
import torch

from .x1_amp_exp11_env import X1AMPExp11Env


class X1AMPExp3Env(X1AMPExp11Env):
    def _reward_tracking_lin_vel(self):
        error = (self.commands[:, :2]-self.base_lin_vel[:, :2]).square().sum(-1)
        return torch.exp(-error/self.cfg.rewards.tracking_lin_sigma)
