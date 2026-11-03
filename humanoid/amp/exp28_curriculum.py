"""Preserve balance and swing; ramp only the new forward correction cost."""
from .exp27_curriculum import Exp27Curriculum


class Exp28Curriculum(Exp27Curriculum):
    def __init__(self):
        super().__init__()
        self.run_start_iteration = 6300

    @property
    def balance_multiplier(self):
        return 1.

    @property
    def forward_multiplier(self):
        return min(1., max(0., (self.last_iteration-self.run_start_iteration)/200.))
