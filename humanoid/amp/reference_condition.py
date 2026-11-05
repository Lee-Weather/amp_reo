"""Explicit finite-reference conditions applied online to the newest frame only."""
import hashlib
import json

import numpy as np


def causal_speed_envelope(velocity, window=21, deadband=.05):
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 1 or not np.isfinite(velocity).all() or window < 1 or deadband < 0:
        raise ValueError('Invalid reference velocity envelope')
    sums = np.r_[0., np.cumsum(np.maximum(velocity, 0))]
    index = np.arange(len(velocity))
    start = np.maximum(0, index-window+1)
    result = (sums[index+1]-sums[start])/(index-start+1)
    result[result <= deadband] = 0.
    return result.astype(np.float32)


class ReferenceCondition:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        self.meta = json.loads(str(self.data.pop('metadata')))
        if self.meta['schema'] != 'x1_reference_progress_condition_v1':
            raise ValueError('Unknown reference condition schema')
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.frames = len(self.data['commands'])
        if self.frames < 2 or self.data['commands'].shape != (self.frames, 3):
            raise ValueError('Invalid reference command shape')
        if self.data['features'].shape != (self.frames, 2):
            raise ValueError('Invalid reference progress shape')
        for key, value in self.data.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite reference condition: '+key)
        expected = np.arange(self.frames)/(self.frames-1)
        if not np.allclose(self.data['progress'], expected, rtol=0, atol=1e-7):
            raise ValueError('Progress must cover the finite reference exactly')
        features = np.stack((np.sin(np.pi*expected), np.cos(np.pi*expected)), axis=-1)
        if not np.allclose(self.data['features'], features, rtol=0, atol=1e-6):
            raise ValueError('Progress features do not match their declared half-circle encoding')

    def sample(self, step):
        if not isinstance(step, (int, np.integer)) or step < 0 or step >= self.frames:
            raise ValueError('Finite reference ended; no implicit hold or loop')
        return self.data['commands'][step].copy(), self.data['features'][step].copy()

    def apply(self, env, step):
        """Playback adapter boundary, before consumer input capture; never edit past frames."""
        import torch
        if abs(env.dt-self.meta['policy_dt']) > 1e-7 or not torch.all(env.episode_length_buf == step).item():
            raise ValueError('Reference progress must use the actual episode control-step clock')
        command, features = self.sample(step)
        command = torch.as_tensor(command, device=env.device, dtype=env.commands.dtype)
        features = torch.as_tensor(features, device=env.device, dtype=env.obs_buf.dtype)
        env.commands[:, :3] = command
        width = env.cfg.env.num_single_obs
        env.obs_history[-1][:, :3] = command*env.commands_scale
        env.obs_history[-1][:, -2:] = features
        env.obs_buf[:, -width:-width+3] = command*env.commands_scale
        env.obs_buf[:, -2:] = features
        if env.privileged_obs_buf is not None:
            env.privileged_obs_buf[:, :3] = command*env.commands_scale
            env.privileged_obs_buf[:, width-2:width] = features
        return env.obs_buf
