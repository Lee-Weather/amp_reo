"""Uncapped corrective costs retain signal beyond the tracking reward's peak."""
from .exp20_rewards import huber

def corridor_costs(lateral_velocity,yaw_rate,yaw_command):
    return huber(lateral_velocity/.15),huber((yaw_rate-yaw_command)/.5)
