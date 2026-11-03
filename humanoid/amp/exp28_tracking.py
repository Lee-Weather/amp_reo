"""Continuous corrective signal for forward overshoot and underspeed."""
from .exp20_rewards import huber


def forward_cost(velocity, command):
    return huber((velocity-command)/.2)
