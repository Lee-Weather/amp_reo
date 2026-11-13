"""exp3.4 env: zero-command exposure diagnostics, with an optional pipeline fix.

``cfg.env.zero_exposure_fix`` selects the behaviour:

* ``False`` — reproduce exp3.3 exactly, but with the new metrics.  Used for the
  diagnostic smoke that shows where the 50% zero draws get lost.
* ``True``  — every periodic resample draws fresh commands itself, so the zero
  share set by ``cfg.env.zero_curriculum`` really reaches
  ``standing_commands.requested`` instead of replaying the reset snapshot or the
  forced native speed of the nominal cohort.
"""
import json

import torch

from humanoid.amp.nominal_startup import cohort_mask
from .x1_amp_exp33_env import X1AMPExp33Env


class X1AMPExp34Env(X1AMPExp33Env):
    """Deployment pd_stand pose (FK-verified) plus zero-command instrumentation."""

    # The exp2.x base class that carries ``_iteration`` is not in this task's MRO,
    # so the learning iteration is kept here and fed by the runner through
    # ``set_training_iteration``.
    _iteration = 0

    def set_training_iteration(self, iteration):
        if iteration < 0:
            raise ValueError("Negative exp3.4 iteration")
        self._iteration = int(iteration)

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

    def _draw_commands(self, count):
        """Draw native-speed commands with a real zero share.

        The inherited sampler (``humanoid/amp/stationary.py:17``) intentionally
        skips zero sampling when ``command_ranges['lin_vel_x']`` is degenerate,
        and this task pins that range to the native speed — so exp3.1/3.2/3.3
        never actually saw a zero-command goal even though ``zero_probability``
        was 0.5.  With ``env.zero_exposure_fix`` the zero share is drawn here
        instead, which also covers the reset-time draw in ``reset_idx``.
        """
        if not getattr(self.cfg.env, "zero_exposure_fix", False):
            return super()._draw_commands(count)
        result = torch.zeros(count, 3, device=self.device)
        result[:, 0] = self.cfg.commands.native_speed
        result[torch.rand(count, device=self.device) < self._zero_probability(), 0] = 0.0
        return result

    def _resample_commands(self, env_ids):
        if not len(env_ids):
            return
        inside_reset = self._reset_reference is not None
        super()._resample_commands(env_ids)
        requested = self.standing_commands.requested[env_ids]

        stats = self._resample_stats
        stats["drawn"] += int(len(env_ids))
        stats["zero"] += int((requested[:, 0].abs() < 0.01).sum())
        if inside_reset:
            stats["from_reference"] += int(len(env_ids))

        drawn = stats["drawn"]
        if drawn:
            self._zero_ratio = stats["zero"] / drawn
            self._reference_ratio = stats["from_reference"] / drawn

        if getattr(self, "_resample_debug", 0) < 12:
            self._resample_debug = getattr(self, "_resample_debug", 0) + 1
            nominal = cohort_mask(env_ids, self.num_envs, 0.25)
            print("ZERO_PIPELINE_DEBUG " + json.dumps({
                "call": self._resample_debug,
                "iteration": int(getattr(self, "_iteration", -1)),
                "zero_p": self._zero_probability(),
                "fix": bool(getattr(self.cfg.env, "zero_exposure_fix", False)),
                "batch": int(len(env_ids)),
                "inside_reset": bool(inside_reset),
                "requested_zero": int((requested[:, 0].abs() < 0.01).sum()),
                "reset_draw_zero": int((self._reset_commands[:, 0].abs() < 0.01).sum()) if self._reset_commands is not None else None,
                "cum_zero": int(stats["zero"]),
                "cum_drawn": int(stats["drawn"]),
            }), flush=True)

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