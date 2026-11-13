"""exp3.4 gates: standing-pose kinematics and the zero-command draw.

The kinematic checks run anywhere (pure numpy + lxml).  The environment checks
need torch/Isaac Gym and are skipped on a plain workstation.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "amp_tools"))

import check_pose_fk as fk  # noqa: E402


def _pose(name="pd_stand"):
    return dict(zip(fk.DOF_NAMES, fk.POSE_PRESETS[name]))


@pytest.fixture(scope="module")
def baseline():
    return fk.evaluate(fk.DEFAULT_URDF, _pose("pd_stand"))


def test_stand_pose_is_symmetric(baseline):
    checks = baseline["checks"]
    assert checks["foot_height_gap_mm"] <= 5.0
    assert checks["foot_front_gap_mm"] <= 10.0


def test_stand_pose_centre_of_mass_inside_support(baseline):
    assert baseline["support"]["com_inside"] is True
    assert baseline["checks"]["com_margin_mm"] >= 20.0


def test_stand_pose_ground_contact_height(baseline):
    # The sole sits 0.611 m below base_link; exp3.4 spawns there instead of 0.64 m.
    assert baseline["base_height_for_contact_m"] == pytest.approx(0.611, abs=0.005)


def test_mirrored_right_hip_pitch_breaks_the_stance():
    flipped = fk.evaluate(fk.DEFAULT_URDF, {**_pose("pd_stand"), "right_hip_pitch_joint": 0.40})
    assert flipped["checks"]["foot_height_gap_mm"] > 20.0
    assert flipped["checks"]["foot_front_gap_mm"] > 50.0


def test_trajectory_mean_is_not_a_static_stance():
    mean = fk.evaluate(fk.DEFAULT_URDF, _pose("mean29"))
    assert mean["support"]["com_inside"] is False


def test_pd_stand_angle_conventions():
    pose = _pose("pd_stand")
    assert pose["right_hip_pitch_joint"] == pytest.approx(-pose["left_hip_pitch_joint"])
    assert pose["right_knee_pitch_joint"] == pytest.approx(pose["left_knee_pitch_joint"])
    assert pose["right_ankle_pitch_joint"] == pytest.approx(pose["left_ankle_pitch_joint"])
    assert pose["right_hip_roll_joint"] == pytest.approx(-pose["left_hip_roll_joint"])
    assert pose["right_hip_yaw_joint"] == pytest.approx(-pose["left_hip_yaw_joint"])


def _env_module():
    try:
        from humanoid.envs.x1.x1_amp_exp34_env import X1AMPExp34Env
        from humanoid.envs.x1.x1_amp_exp34_config import X1AMPExp34Cfg
    except Exception as error:  # torch/Isaac Gym are unavailable off the training box
        pytest.skip(f"exp3.4 env not importable here: {error}")
    return X1AMPExp34Env, X1AMPExp34Cfg


def test_zero_flag_curriculum_and_spawn_height():
    _, cfg = _env_module()
    assert cfg.env.zero_exposure_fix is True
    assert cfg.env.zero_curriculum == (100, 0.3, 0.5)
    assert cfg.env.standing_pose_z == pytest.approx(0.611)


def test_env_pose_matches_config_pose():
    env_cls, cfg = _env_module()
    assert list(env_cls._PD_STAND_ANGLES) == pytest.approx(fk.POSE_PRESETS["pd_stand"])
    angles = list(cfg.init_state.default_joint_angles.values())
    assert angles == pytest.approx(fk.POSE_PRESETS["pd_stand"])


def test_draw_commands_mixes_native_and_zero():
    env_cls, _ = _env_module()
    stub = SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(env=SimpleNamespace(zero_exposure_fix=True),
                            commands=SimpleNamespace(native_speed=0.5953696966)),
        _zero_probability=lambda: 0.5,
    )
    commands = env_cls._draw_commands(stub, 20000)
    zero_share = float((commands[:, 0].abs() < 1e-6).float().mean())
    native_share = float((commands[:, 0] > 0.99 * 0.5953696966).float().mean())
    assert zero_share == pytest.approx(0.5, abs=0.02)
    assert native_share == pytest.approx(0.5, abs=0.02)