"""exp3.3 env: standard-upright standing pose for zero-command."""
import torch

from .x1_amp_exp32_env import X1AMPExp32Env


class X1AMPExp33Env(X1AMPExp32Env):
    """Deployment standard-upright standing for zero-command anchor.

    Overrides the standing_state from the lean-forward stand_to_walk clip
    (frame 11) to the deployment pd_stand upright pose.

    pd_stand init_state (deployment):
      left/right shoulders pitch 0.15, roll -0.10, elbow 0.30
      left hip  0.40 / -0.31 /  0.05, knee 0.49, ankle -0.21
      right hip -0.40 /  0.31 / -0.05, knee 0.49, ankle -0.21
      lumbar = neutral (0), all aux joints (shoulder_yaw, elbow_yaw, wrist, ankle_roll) = 0
    """

    # ── pd_stand joint angles (29 values, DOF_NAMES order) ─────────
    _PD_STAND_ANGLES = [
        # lumbar: yaw, roll, pitch – neutral
        0.00, 0.00, 0.00,
        # left arm: pitch, roll, yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll
        0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
        # right arm: pitch, roll, yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll
        0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
        # left leg: hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
        0.40, 0.05, -0.31, 0.49, -0.21, 0.00,
        # right leg: hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
        -0.40, -0.05, 0.31, 0.49, -0.21, 0.00,
    ]

    def _init_buffers(self):
        """Initialise buffers, then override standing_state with pd_stand."""
        super()._init_buffers()

        # ── Build pd_stand standing_state vector (71 elements) ─────
        # [0:3]   base position
        pos = self.standing_state.new_tensor([0.0, 0.0, 0.64])

        # [3:7]   base orientation – identity (upright, no tilt)
        quat = self.standing_state.new_tensor([1.0, 0.0, 0.0, 0.0])

        # [7:13]  base linear & angular velocity – zero
        vel = self.standing_state.new_zeros(6)

        # [13:42] joint angles
        angles = self.standing_state.new_tensor(self._PD_STAND_ANGLES)

        # [42:71] joint velocities – zero
        jvel = self.standing_state.new_zeros(29)

        state = torch.cat([pos, quat, vel, angles, jvel])

        self.standing_state = state

        # Update dependent quantities
        # identity quaternion → gravity is already [0, 0, -1] with no rotation
        gravity = state.new_tensor([0.0, 0.0, -1.0])
        self.standing_gravity = gravity
