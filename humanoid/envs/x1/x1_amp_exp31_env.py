"""exp3.1: standing stability reward."""
from .x1_amp_exp30_env import X1AMPExp30Env


class X1AMPExp31Env(X1AMPExp30Env):
    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        ang_scale = getattr(cfg, 'standing_stability_angular_scale', 0.)
        ang_max = getattr(cfg, 'standing_stability_max_angular', 0.15)

        if ang_scale < 0 and ang_max > 0:
            standing = self.commands[:, 0].abs() < 0.05
            angular = self.base_ang_vel.norm(dim=-1).clamp(max=float(ang_max))
            penalty = float(ang_scale) * (angular / ang_max) * standing.float()
            self.rew_buf.add_(penalty * self.dt)
            self.extras.setdefault("training_metrics", {}).update(
                StandingStability_penalty=penalty,
                StandingStability_angular=angular * standing.float(),
            )
