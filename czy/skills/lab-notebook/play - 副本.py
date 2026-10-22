# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.


import os
import csv
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from humanoid.utils.evaluation import inject_evaluation_command
from humanoid.utils.playback_diagnostics import capture_playback_step
from isaacgym.torch_utils import *

import torch
from datetime import datetime

import pygame
from threading import Thread


x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
joystick_use = True
joystick_opened = False

# Unified playback: six stages, each lasting 10 simulation seconds.
PLAY_SPEEDS = (0.0, 0.4, 0.6, 0.0, 0.4, 0.0)
PLAY_STAGE_SECONDS = 10.0


def current_command(step_idx, profile):
    """返回控制步 step_idx 对应的 command_x。"""
    acc = 0
    for steps, vel in profile:
        if step_idx < acc + steps:
            return vel
        acc += steps
    return 0.0


def current_stage(step_idx, profile):
    end = 0
    for index, (steps, _) in enumerate(profile):
        end += steps
        if step_idx < end:
            return index
    return len(profile) - 1
# ==============================================================================

if joystick_use:
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    zero_residual = getattr(args, "zero_residual", False)
    if zero_residual:
        if not hasattr(env_cfg.control, "residual_enabled"):
            raise ValueError("--zero_residual requires a reference-residual task")
        if args.checkpoint_url_b64 is not None or args.ckpt_path is not None or args.resume:
            raise ValueError("zero-residual playback does not load a checkpoint")
        env_cfg.control.residual_enabled = False
        env_cfg.domain_rand.add_lag = False
        env_cfg.domain_rand.add_dof_lag = False
        env_cfg.domain_rand.add_dof_pos_vel_lag = False
        env_cfg.domain_rand.add_imu_lag = False
        env_cfg.noise.add_noise = False
    from humanoid.utils.evaluation import configure_evaluation
    mode = args.eval_mode
    if mode is None:
        mode = "nominal_no_push" if getattr(env_cfg.rewards, "reference_pose_tracking", False) else (
            "train" if os.environ.get("PLAY_MATCH_TRAIN", "1") != "0" else "nominal_no_push")
    seed = args.seed if args.seed is not None else 123145
    configure_evaluation(env_cfg, mode, seed)
    train_cfg.seed = seed
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    if RENDER:
        env_cfg.env.enable_headless_render = True
    _play_friction = os.environ.get("PLAY_FRICTION")
    if _play_friction is not None:
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [float(_play_friction)] * 2
        env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    print("[play] eval_mode={}, seed={}, pushes={}, noise={}".format(
        mode, seed, env_cfg.domain_rand.push_robots, env_cfg.noise.add_noise))

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)


    if RENDER and env.viewer is not None:
        # 仅录制模式需要相机视角（set_camera 依赖 viewer，headless 下不可调用）
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)


    # ---- 云端回放模式：--checkpoint_url_b64 = 签名下载 URL 的 URL-safe Base64 ----
    # 约定：checkpoint 与 model_isaac_csv.pt 都落在 logs/<experiment_name>/gm_play/（SDK 已识别的 PT 目录）
    gm_mode = getattr(args, "checkpoint_url_b64", None) is not None
    gm_play_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'gm_play')

    if gm_mode:
        import base64 as _b64
        import re as _re
        import urllib.request as _urlreq
        os.makedirs(gm_play_dir, exist_ok=True)
        ckpt_url = _b64.urlsafe_b64decode(args.checkpoint_url_b64.encode()).decode()
        m = _re.search(r"model_(\d+)\.pt", ckpt_url)
        ckpt_num = int(m.group(1)) if m else 0
        gm_ckpt_path = os.path.join(gm_play_dir, f"model_{ckpt_num}.pt")
        if not os.path.exists(gm_ckpt_path):
            print(f"[gm] downloading checkpoint -> {gm_ckpt_path}")
            _urlreq.urlretrieve(ckpt_url, gm_ckpt_path)
        print(f"[gm] checkpoint ready: {gm_ckpt_path}")

    # load policy
    if zero_residual:
        from types import SimpleNamespace
        ppo_runner = SimpleNamespace(alg=None)
        env.reset()
        policy = lambda observations: torch.zeros(env.num_envs, env.num_actions, device=env.device)
    elif gm_mode:
        # 下载路径不在 exported_data 下，绕过 make_alg_runner 内部 resume，手动加载
        train_cfg.runner.resume = False
        ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        ppo_runner.load(gm_ckpt_path, load_optimizer=False)
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if not zero_residual:
        policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY and not zero_residual:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        if getattr(env_cfg.env, "observe_reference_pose", False):
            from humanoid.scripts.export_policy_dh import ExportedDH
            from humanoid.utils.residual_control import export_residual_control_spec
            actor = ppo_runner.alg.actor_critic
            os.makedirs(path, exist_ok=True)
            ExportedDH(actor.actor, actor.long_history, actor.state_estimator,
                       actor.num_short_obs, actor.in_channels, actor.num_proprio_obs).export(
                           os.path.join(path, "policy_dh.jit"))
            export_residual_control_spec(env_cfg, path)
        else:
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    dt = env_cfg.sim.dt * env_cfg.control.decimation
    video_fps = 1.0 / (2 * dt)
    profile = [(max(1, round(PLAY_STAGE_SECONDS / dt)), speed) for speed in PLAY_SPEEDS]
    total_play_steps = sum(steps for steps, _ in profile)
    run_name_str = args.run_name if args.run_name is not None else "test"
    logger = Logger(dt)
    robot_index = 0 # which robot is used for logging
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        # camera_properties.width = 1280   # 原值: 1920
        # camera_properties.height = 720   # 原值: 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        # camera_offset = gymapi.Vec3(1, -1, 0.5)
        # 修改视角把 Z 从 0.5 提高到 1.5，同时把 X,Y 距离拉大到 2.0
        camera_offset = gymapi.Vec3(2.0, -2.0, 1.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 视频/CSV 输出到 logs/<experiment_name>/play_output/
        run_name_str = args.run_name if args.run_name is not None else "test"
        custom_save_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'play_output')
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name_str}.mp4"
        video_filepath = os.path.join(custom_save_path, file_name)

        # 如果文件夹不存在，自动创建
        if not os.path.exists(custom_save_path):
            os.makedirs(custom_save_path, exist_ok=True)

        print(f"Recording video to: {video_filepath}")
        # One recorded frame per two control steps.
        video = cv2.VideoWriter(video_filepath, fourcc, video_fps, (1920, 1080))
        if not video.isOpened():
            raise RuntimeError("cannot open video writer: " + video_filepath)
        # video = cv2.VideoWriter(video_filepath, fourcc, 25.0, (1280, 720))


        # video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        # experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        # dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        # if not os.path.exists(video_dir):
        #     os.makedirs(video_dir,exist_ok=True)
        # if not os.path.exists(experiment_dir):
        #     os.makedirs(experiment_dir,exist_ok=True)
        # video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    
    obs = env.get_observations()
    frame_count = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    # 足部刚体索引（用于诊断与视频叠加的接触力，替代硬编码索引）

    # 诊断输出目录（CSV 不依赖 RENDER，始终输出）
    diag_out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'play_output')
    os.makedirs(diag_out_dir, exist_ok=True)
    # 对齐真机 walk_diag 列（czy/real_data/13rt/walk_diag_*.csv）：基座 + 逐关节 action/pos/vel/effort/pos_des_raw
    # 保留 base_vel_x/y/z 旧列（isaac-diag-eval 与既有分析依赖）
    diag = {}
    clip_count = 0  # 力矩限幅累计计数（对应真机 clip_count）

    # =========== 新增：初始化速度累加器 ===========
    vel_sum = 0.0       # 速度总和
    step_accum = 0      # 步数计数器
    # ===========================================

    # §21b: reset 后站立缓冲（控制步数）。训练中 reset→stand 段(3~5s)→walk 段切指
    # 令（"站稳再走"）；play 若 reset 后立即注入当前段全速指令，属分布外空中起步
    # （h=0.699 悬空、双脚离地 0.13m 落地即迈步）——实测左脚先摆→roll 单调右漂
    # 3~4s 倒（lagfix 回放 3/3 段复现）。缓冲期命令置 0，对齐训练渐进。
    STAND_BUFFER_STEPS = round(1.5 / dt)
    stand_buffer = 0

    for i in range(total_play_steps):

        requested = current_command(i, profile) if FIX_COMMAND else x_vel_cmd
        applied = 0.0 if stand_buffer > 0 else requested
        obs = inject_evaluation_command(env, applied)
        if not FIX_COMMAND:
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.obs_buf[:, -env_cfg.env.num_single_obs + 3:-env_cfg.env.num_single_obs + 5] = (
                env.commands[:, 1:3] * env.commands_scale[1:3])
        actions = policy(obs.detach())
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        # §21b: env0 reset 触发站立缓冲倒计时
        if dones[robot_index]:
            stand_buffer = STAND_BUFFER_STEPS
        elif stand_buffer > 0:
            stand_buffer -= 1
        row = capture_playback_step(env, infos, rews, dones, ppo_runner.alg,
                                    requested, current_stage(i, profile), clip_count, robot_index)
        clip_count = int(row["clip_count"])
        for name, value in row.items():
            diag.setdefault(name, []).append(value)
        current_vel_x = row["base_vel_x"]
        real_cmd_x = row["command_x"]
        vel_sum += current_vel_x
        step_accum += 1
        if RENDER:
            frame_count += 1
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)

            if frame_count % 2 == 0:
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                # img = np.reshape(img, (720, 1280, 4))
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # # ==================== 添加前进速度记录 ====================

                # # 1. 获取数据
                # target_vel = env.commands[0, 0].item()
                
                # # 计算平均速度 (防止除以0)
                # avg_vel = vel_sum / step_accum if step_accum > 0 else 0.0
                
                # # 2. 准备显示的文本 (稍微长一点)
                # # 格式：CMD(指令) | REAL(瞬时) | AVG(平均)
                # info_text = f"CMD: {target_vel:.2f} | REAL: {current_vel_x:.2f} | AVG: {avg_vel:.2f}"
                
                # # 3. 计算文字位置
                # # 因为文字变长了，为了不跑出画面，我们需要把起始位置往左移
                # img_h, img_w = img.shape[:2]
                # text_pos = (img_w - 950, 60)  # 从 -550 改为 -750，留出更多空间

                # # 4. 绘制文字 (黑边 + 青字)
                # cv2.putText(img, info_text, text_pos, 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                # cv2.putText(img, info_text, text_pos, 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                # # ==================== 前进速度结束 ====================
                # ==================== 1. 获取基础数据 ====================
                # 速度数据
                target_vel = diag["command_applied"][-1]
                current_vel_x = diag["base_vel_x"][-1]
                avg_vel = vel_sum / step_accum if step_accum > 0 else 0.0

                # y/z 方向与偏航速度（诊断缓冲区最新值）
                current_vel_y = diag["base_vel_y"][-1]
                current_vel_z = diag["base_vel_z"][-1]
                current_vel_yaw = diag["base_ang_vel_z"][-1]

                # 接触力数据（使用 feet_indices，与诊断一致）
                left_force = diag["foot_force_l"][-1]
                right_force = diag["foot_force_r"][-1]
                
                # 接触判断 (阈值 1.0 N)
                l_on = left_force > 1.0
                r_on = right_force > 1.0

                # ==================== 2. 定义显示布局 ====================
                img_h, img_w = img.shape[:2]
                base_x = img_w - 1150  # 起始 X 坐标
                base_y = 60           # 起始 Y 坐标
                line_height = 50      # 行高

                # 辅助函数：快速绘制带描边的文字
                def draw_outlined_text(image, text, pos, color, scale=0.9):
                    # 黑描边
                    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
                    # 彩色字
                    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

                # ==================== 3. 绘制第一行：x 速度信息 ====================
                speed_text = f"CMD: {target_vel:.2f} | REAL: {current_vel_x:.2f} | AVG: {avg_vel:.2f}"
                draw_outlined_text(img, speed_text, (base_x, base_y), (255, 255, 0), 1.0) # 青色

                # ==================== 3.1 绘制第二行：y/z 方向与偏航速度 ====================
                vel_yz_text = f"VEL_Y: {current_vel_y:+.2f} | VEL_Z: {current_vel_z:+.2f} | VEL_YAW: {current_vel_yaw:+.2f}"
                draw_outlined_text(img, vel_yz_text, (base_x, base_y + line_height), (0, 255, 255), 0.9) # 黄色

                # ==================== 4. 绘制第三、四行：单脚状态 ====================
                # 左脚
                l_color = (0, 255, 0) if l_on else (0, 0, 255) # 绿/红
                l_text = f"L-FOOT: {'ON ' if l_on else 'OFF'} ({left_force:.1f} N)"
                draw_outlined_text(img, l_text, (base_x, base_y + line_height * 2), l_color)

                # 右脚
                r_color = (0, 255, 0) if r_on else (0, 0, 255) # 绿/红
                r_text = f"R-FOOT: {'ON ' if r_on else 'OFF'} ({right_force:.1f} N)"
                draw_outlined_text(img, r_text, (base_x, base_y + line_height * 3), r_color)

                # ==================== 5. 绘制第五行：步态全局状态 (新增) ====================
                
                state_text = "STATE: SINGLE SUPPORT" # 默认单支撑
                state_color = (200, 200, 200)        # 默认灰色

                if l_on and r_on:
                    # 双脚着地 (Double Support)
                    state_text = "STATE: *** DOUBLE SUPPORT ***"
                    state_color = (0, 255, 255) # 黄色 (BGR: Yellow)
                
                elif not l_on and not r_on:
                    # 双脚离地 (Flight Phase)
                    state_text = "STATE: >>> FLIGHT PHASE <<<"
                    state_color = (255, 0, 255) # 紫色 (BGR: Magenta)

                # 绘制状态
                draw_outlined_text(img, state_text, (base_x, base_y + line_height * 4), state_color, 1.0)

                # ==================== 结束绘制 ====================
               

                video.write(img[..., :3])
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    # =========== 回放结束：写出诊断 CSV + 分段 Summary ===========
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    csv_path = os.path.join(diag_out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name_str}_isaac_diag.csv")
    header = ["step", "time_s"] + list(diag)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(len(diag["command_x"])):
            writer.writerow([step, round(step * dt, 6)] + [diag[name][step] for name in diag])
    import json
    metadata = dict(task=args.task, eval_mode=mode, seed=seed, control_dt=dt,
                    physics_dt=env_cfg.sim.dt, video_fps=video_fps,
                    profile=profile, steps=total_play_steps,
                    checkpoint=args.checkpoint, load_run=args.load_run, zero_residual=zero_residual,
                    checkpoint_path=getattr(ppo_runner, "loaded_checkpoint_path", None),
                    resets=sum(diag["done"]), timeouts=sum(diag["timeout"]),
                    success=not any(diag["done"]), torque_clip_count=clip_count,
                    pushes=env_cfg.domain_rand.push_robots, noise=env_cfg.noise.add_noise)
    with open(csv_path.replace("_isaac_diag.csv", "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved diagnostic CSV -> {csv_path}")
    print(f"Torque clip count (joint saturation events across physics substeps): {clip_count}")

    print("\n===== Speed Profile Summary =====")
    acc = 0
    for seg_i, (steps, vel) in enumerate(profile):
        seg_vels = diag["base_vel_x"][acc:acc + steps]
        print(f"  Segment {seg_i}: cmd={vel:.2f} m/s | avg_real={np.mean(seg_vels):.3f} m/s")
        acc += steps

    if RENDER:
        video.release()

    # ---- 云端回放（gm）模式产物打包：视频平铺 + CSV 打包为 model_isaac_csv.pt ----
    if gm_mode:
        import shutil
        import time as _time
        if RENDER and os.path.exists(video_filepath):
            flat_video = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'play_output.mp4')
            shutil.copyfile(video_filepath, flat_video)
            print(f"[gm] video copied -> {flat_video}")
        pack_path = os.path.join(gm_play_dir, "model_isaac_csv.pt")
        with open(csv_path, "rb") as f:
            torch.save({"bytes": f.read(), "filename": os.path.basename(csv_path)}, pack_path)
        print(f"[gm] packed CSV -> {pack_path}")
        print("[gm] keeping files for SDK scan/upload (60s)...")
        _time.sleep(60)

if __name__ == '__main__':
    EXPORT_POLICY = False
    # exp1.6 回放 workaround（2026-09-09）：本机 Vulkan/相机层段错误（×2 复现，CUDA 正常），
    # 临时经环境变量 PLAY_RENDER=0 关渲染先出 CSV；恢复视频再置 1 或走云端 gm_mode 回放
    RENDER = os.environ.get("PLAY_RENDER", "1") != "0"
    FIX_COMMAND = True
    args = get_args()
    play(args)
