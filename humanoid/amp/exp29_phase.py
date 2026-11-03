"""Command-scaled phase clock shared by training and MuJoCo replay."""
import torch


def phase_speed_scale(command, native_speed):
    if native_speed <= 0:
        raise ValueError('Native speed must be positive')
    return (torch.as_tensor(command)/native_speed).clamp(0, 1)


def advance_phase(phase, command, dt, period, native_speed):
    return (phase + dt/period*phase_speed_scale(command, native_speed)).remainder(1)
