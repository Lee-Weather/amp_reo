"""Continue all learned strengths; the only change is transition reference time."""
from .exp28_curriculum import Exp28Curriculum


class Exp29Curriculum(Exp28Curriculum):
    def __init__(self):
        super().__init__()
        self.run_start_iteration = 7500

    @property
    def forward_multiplier(self):
        return 1.
