"""exp3.2 env: fix zero-command exposure, bank-observation injection."""
import torch
from .x1_amp_exp31_env import X1AMPExp31Env


class X1AMPExp32Env(X1AMPExp31Env):
    """Fix zero-command propagation to nominal envs.

    The base _resample_commands (x1_amp_env.py:871) unconditionally overrides
    nominal-env commands to [native_speed, 0, 0], which drowns the 50% zero-
    command sampling.  This override removes that line so zero commands reach
    the policy observation.

    Also logs zero-requested/applied fractions for monitoring.
    """

    def _resample_commands(self, env_ids):
        """Draw fresh commands, keeping zero-command samples."""
        nominal = self._nominal(env_ids)
        if self._reset_commands is not None:
            self._reset_commands[nominal] = self.commands.new_tensor(
                [self.cfg.commands.native_speed, 0., 0.])
        requested = (self._reset_commands if self._reset_commands is not None
                     else self._draw_commands(len(env_ids)))
        self.standing_commands.requested[env_ids] = requested

        # ════════════════════════════════════════════════════════════
        # FIX:  no longer override nominal-env commands to native_speed
        # ════════════════════════════════════════════════════════════

        if self._reset_reference is not None or not self._inside_step:
            self.commands[:, :3] = self.standing_commands.applied()

        # ── logging ────────────────────────────────────────────────
        applied = self.standing_commands.applied()
        self.extras.setdefault("training_metrics", {}).update({
            "zero_requested_raw": (
                requested[:, 0].abs() < 0.01).float().mean(),
            "zero_applied_raw": (
                applied[:, 0].abs() < 0.01).float().mean(),
        })

    def compute_reward(self):
        """Append zero-command tracking metrics."""
        super().compute_reward()
        applied = self.standing_commands.applied()
        self.extras.setdefault("training_metrics", {}).update({
            "zero_applied_fraction": (
                applied[:, 0].abs() < 0.05).float().mean(),
        })

    def generate_standing_observations(self, policy, num_obs=256):
        """Step the env with zero commands and collect observations.

        Returns a tensor of shape (num_obs, obs_dim) that can be used as an
        anchor bank for policy regularisation / hard rejection.

        Must be called AFTER a full env reset (e.g. at warm_start) so the env
        is in a consistent state.  The caller is responsible for restoring the
        env afterward.
        """
        self.reset()
        # force zero command on all envs
        self.commands[:, :3] = 0.
        self.standing_commands.requested[:] = 0.
        self.standing_commands.value[:] = 0.
        self.standing_commands.active_goal[:] = 0.

        obs_list = []
        with torch.no_grad():
            for _ in range(num_obs // max(1, self.num_envs) + 1):
                self.compute_observations()
                obs_list.append(self.obs_buf[:num_obs].clone())
                actions = policy(self.obs_buf)
                self.step(actions)

        return torch.cat(obs_list, dim=0)[:num_obs]
