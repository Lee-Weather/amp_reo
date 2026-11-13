"""exp3.5 env: exp3.4's zero-command pipeline with the exp3.2 standing frame."""
from .x1_amp_exp32_env import X1AMPExp32Env
from .x1_amp_exp34_env import X1AMPExp34Env


class X1AMPExp35Env(X1AMPExp34Env):
    """Keep the zero-command draw and metrics, restore the motion-clip stand pose."""

    def _init_buffers(self):
        # Start the MRO at exp3.2 so exp3.3/exp3.4's pd_stand override is skipped;
        # exp3.2 itself has no ``_init_buffers``, so this runs the exp1.x baseline
        # that builds ``standing_state`` from the motion library's stand frame.
        X1AMPExp32Env._init_buffers(self)
        self._resample_stats = {"drawn": 0, "zero": 0, "from_reference": 0}
        self._zero_ratio = 0.0
        self._reference_ratio = 0.0