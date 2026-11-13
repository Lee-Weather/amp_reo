"""exp3.4 env: zero-command exposure diagnostics, with an optional pipeline fix.

``cfg.env.zero_exposure_fix`` selects the behaviour:

* ``False`` — reproduce exp3.3 exactly, but with the new metrics.  Used for the
  diagnostic smoke that shows where the 50% zero draws get lost.
* ``True``  — every periodic resample draws fresh commands itself, so the zero
  share set by ``cfg.env.zero_curriculum`` really reaches
  ``standing_commands.requested`` instead of replaying the reset snapshot or the
  forced native speed of the nominal cohort.
"""
import torch

from .x1_amp_exp33_env import X1AMPExp33Env


class X1AMPExp34Env(X1AMPExp33Env):
    """Deployment pd_stand pose (FK-verified) plus zero-command instrumentation."""

    def _init_buffers(self):
        super()._init_buffers()
        # FK on X1_29DOF_physically_mirrored.urdf: sole touches the ground at 0.611 m.
        self.standing_state[2] = float(getattr(self.cfg.env, "standing_pose_z", 0.611))
        self._resample_stats = {"drawn": 0, "zero": 0, "from_reference": 0}
        self._zero_ratio = 0.0
        self._reference_ratio = 0.0

    # ── command sampling ────────────────────────────────────────────────
    def _zero_probability(self):
        warmup, low, high = getattr(self.cfg.env, "zero_curriculum", (0, 0.5, 0.5))
        iteration = int(getattr(self, "_iteration", 0))
        return float(low if iteration < warmup else high)

    def _draw_commands_fresh(self, count):
        result = torch.zeros(count, 3, device=self.device)
        result[:, 0] = self.cfg.commands.native_speed
        zero = torch.rand(count, device=self.device) < self._zero_probability()
        result[zero, 0] = 0.0
        return result

    def _resample_commands(self, env_ids):
        if not len(env_ids):
            return
        inside_reset = self._reset_reference is not None
        if inside_reset or not getattr(self.cfg.env, "zero_exposure_fix", False):
            # exp3.3 behaviour: inside a reset the reset-time draw is kept, outside
            # the exp3.2 override decides (it may replay the snapshot or force the
            # nominal cohort to native speed).
            super()._resample_commands(env_ids)
            requested = self.standing_commands.requested[env_ids]
        else:
            requested = self._draw_commands_fresh(len(env_ids))
            self.standing_commands.requested[env_ids] = requested

        stats = self._resample_stats
        stats["drawn"] += int(len(env_ids))
        stats["zero"] += int((requested[:, 0].abs() < 0.01).sum())
        if inside_reset:
            stats["from_reference"] += int(len(env_ids))

        drawn = stats["drawn"]
        if drawn:
            self._zero_ratio = stats["zero"] / drawn
            self._reference_ratio = stats["from_reference"] / drawn

    # ── metrics ─────────────────────────────────────────────────────────
    def compute_reward(self):
        super().compute_reward()
        metrics = self.extras.setdefault("training_metrics", {})
        applied = self.standing_commands.applied()
        steady = self.standing_commands.remaining == 0
        zero = applied[:, 0].abs() < 0.05
        metrics["Commands/zero_applied_fraction"] = (
            zero[steady].float().mean() if bool(steady.any()) else zero.new_zeros(()))
        metrics["Commands/zero_applied_all_fraction"] = zero.float().mean()
        metrics["Commands/stand_buffer_active_fraction"] = (
            self.standing_commands.remaining > 0).float().mean()
        metrics["Commands/zero_requested_fraction"] = applied.new_tensor(self._zero_ratio)
        metrics["Commands/command_from_reference_fraction"] = applied.new_tensor(self._reference_ratio)
        metrics["Commands/resampled_envs"] = applied.new_tensor(float(self._resample_stats["drawn"]))