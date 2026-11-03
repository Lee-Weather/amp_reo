"""Slow transition phase with applied command; keep full-speed reference exact."""
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.amp.exp29_phase import phase_speed_scale, advance_phase
from .x1_amp_exp28_env import X1AMPExp28Env


class X1AMPExp29Env(X1AMPExp28Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum = Exp29Curriculum()

    def _advance_reference_phase(self,active):
        self.gait_phase[active] = advance_phase(self.gait_phase[active],
            self.standing_commands.value[active],self.dt,self.reference_guide.period,self.cfg.commands.native_speed)

    def _blended_reference(self,blend,derivative):
        return self.reference_guide.blended_pose(self.gait_phase,self.standing_state,blend,derivative,
            phase_rate_scale=phase_speed_scale(self.standing_commands.value,self.cfg.commands.native_speed))

    def compute_reward(self):
        super().compute_reward()
        moving = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        rate = phase_speed_scale(self.standing_commands.value,self.cfg.commands.native_speed)/self.reference_guide.period
        self.extras['training_metrics']['Reference/phase_rate_hz'] = rate[moving]

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        result.update(curriculum='command_scaled_phase_low_amp_v1',phase_clock='command_scaled_v1',
            phase_rate='clamp(applied_vx/native_vx,0,1)/native_period',
            transition_dq='blend*phase_scale*native_dq+blend_derivative*(native_q-standing_q)',
            forward_cost_ramp_updates=0)
        return result
