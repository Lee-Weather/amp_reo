"""Keep learned swing/AMP strengths; introduce balance costs over200 updates."""
from .exp26_curriculum import Exp26Curriculum

class Exp27Curriculum(Exp26Curriculum):
    def __init__(self):
        super().__init__()
        self.run_start_iteration=5100

    @property
    def swing_multiplier(self):return 2.

    @property
    def balance_multiplier(self):
        return min(1.,max(0.,(self.last_iteration-self.run_start_iteration)/200.))
