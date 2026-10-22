"""Validate the actual Isaac Gym importer, link frames and auto-reset boundary."""
import isaacgym  # Must precede torch.
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.amp.motion import quat_inverse_rotate


def main():
    args = get_args()
    if args.task != "x1_amp":
        raise ValueError("Use --task=x1_amp")
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs or 8
    cfg.noise.add_noise = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.reset()
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, critic, _, _, info = env.step(action)
    relative = env.rigid_state[:, env.key_body_indices, :3]-env.root_states[:, None, :3]
    quat = env.base_quat[:, None, :].expand(-1, len(env.key_body_indices), -1)
    measured = quat_inverse_rotate(quat, relative)
    expected = env.motion_library.fk.key_positions(env.dof_pos)
    error = (measured-expected).norm(dim=-1).max().item()
    print("SIM_FK_MAX_ERROR_M={:.8f}".format(error), flush=True)
    assert error < 0.002, "Simulator link origins disagree with reference kinematics"
    assert obs.shape == (env.num_envs, 480) and critic.shape == (env.num_envs, 99)
    assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
    # Force a simultaneous timeout to exercise reset indexing and true terminal history.
    env.episode_length_buf[:] = int(env.max_episode_length)
    _, _, _, done, info = env.step(action)
    assert done.all() and info["time_outs"].all()
    assert (env.episode_length_buf == 0).all()
    assert info["amp_obs"].shape == (env.num_envs, 3, 86)
    assert not torch.equal(info["amp_obs"], env.get_amp_observations())
    assert torch.isfinite(env.get_amp_observations()).all()
    _, _, _, _, info = env.step(action)
    assert not info["time_outs"].any(), "Previous step's timeout mask leaked after reset"
    for _ in range(30):
        obs, critic, reward, done, _ = env.step(torch.randn_like(action)*0.1)
        assert all(torch.isfinite(x).all() for x in (obs, critic, reward))
    print("AMP_SIM_VALIDATION_OK dofs=29 obs=480 critic=99 amp=3x86", flush=True)


if __name__ == "__main__":
    main()
