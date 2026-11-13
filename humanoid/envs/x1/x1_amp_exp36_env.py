"""exp3.6 env: standing resets start from walking states (brake-to-stand).

The exp3.x task chain uses ``X1AMPExp11Env.reset_idx`` (the exp2.x ``reset_idx``
with ``_adjust_reset_states`` is not in this MRO), so the reset distribution has
to be changed here rather than in a hook.
"""
import torch

from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp35_env import X1AMPExp35Env


class X1AMPExp36Env(X1AMPExp35Env):
    """Keep the sampled walking state for standing resets instead of a static pose."""

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        lib = self.motion_library
        probability = self.cfg.amp.rsi_probability if self.cfg.amp.reference_state_initialization else 0.
        if not 0 <= probability <= 1:
            raise ValueError("rsi_probability must be in [0,1]")
        standing = torch.rand(len(env_ids), device=self.device) >= probability
        ids, times = lib.sample_ids_times(len(env_ids))
        state = lib.state(ids, times)
        history = lib.history(ids, times)

        # exp3.4/3.5 spawned every standing episode from the static standing pose;
        # on the rounded ankle housing that state has no balance gradient and every
        # attempt topples backwards in 1.4-1.9 s.  Keep the sampled *walking* state
        # for a fraction of them, so the zero command becomes a braking manoeuvre
        # the base policy already knows how to do.
        fraction = float(getattr(self.cfg.env, "moving_stand_fraction", 1.0))
        static = standing & (torch.rand(len(env_ids), device=self.device) >= fraction)
        state[static] = self.standing_state
        noise = self.cfg.amp.standing_joint_noise
        state[static, 13:42] += (2*torch.rand(int(static.sum()), 29, device=self.device)-1)*noise
        state[:, 13:42] = state[:, 13:42].maximum(self.hard_dof_limits[:, 0]).minimum(self.hard_dof_limits[:, 1])
        if bool(static.any()):
            state[static, :3] = lib.grounding.align(
                state[static, :3], state[static, 3:7], state[static, 13:42])
            history[static] = lib.features(state[static])[:, None, :]
        self._reset_reference = state
        self.reset_standing[env_ids] = standing
        self.standing_commands.reset(env_ids, standing)
        LeggedRobot.reset_idx(self, env_ids)
        for frame in self.obs_history:
            frame[env_ids] = 0
        self.amp_history[env_ids] = history
        self._reset_reference = None