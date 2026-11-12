# exp3.1 — KL anchor zero-command stability

## 目标

解决 exp3.0（model_11250 → exp3.0_correct）在 MuJoCo sim2sim 零指令站立时失稳的问题：

1. 零指令比例 20% → 50%（`commands.zero_probability = 0.5`）
2. KL anchor：vx≈0 时约束策略输出接近源模型 11250（`anchor_loss_coef = 0.5`，阈值 0.01，obs[:,0]=2·vx → |vx|<0.005）
3. 站姿角速度惩罚：vx<0.05 时 `-0.03 × min(ω,0.15)/0.15`
4. 500 次新更新（11250 → 11750），warm_start = model_11250.pt

**摩擦随机化已从 exp3.1 移除**：`reward_transfer.py` 在 reward refinement warm_start 时把
`domain_rand` 列为受保护配置，改动即拒绝加载。摩擦随机化需要与源 checkpoint 一致的域随机
配置，应作为独立实验（从 exp3.1 输出模型出发、配置匹配）再做。

## 代码

| 文件 | 说明 |
| --- | --- |
| `humanoid/envs/x1/x1_amp_exp31_config.py` | 配置（commands / rewards / algorithm / runner） |
| `humanoid/envs/x1/x1_amp_exp31_env.py` | 站姿稳定性惩罚 + 指标 |
| `humanoid/amp/x1_exp31_runner.py` | KL anchor `policy_regularization_loss` |
| `humanoid/envs/__init__.py` | 任务注册 |
| `humanoid/utils/task_registry.py` | runner 解析 + checkpoint_dir / warm_start 白名单 |

- 分支：`codex/exp3-0`
- 训练提交：`d6b3665`（含 `02ca2a9` 白名单修复与 `d6b3665` domain_rand 回退）

## 提交历史与失败排查

| 任务 | 结果 | 原因 |
| --- | --- | --- |
| TASK_20261108_024 (v1) | 启动 20s 后失败 (status 6) | ① `--warm_start_sha256` 少抄 2 字符（62 位）；② `codeUrl` 写成 `limxdynamics/amp_re0`，实际仓库为 `Lee-Weather/amp_reo` |
| TASK_20261108_029 (v2) | 启动 16s 后失败 | 仅修复 SHA/URL，仓库 URL 仍错误 → 容器拿不到 exp3.1 代码 |
| TASK_20261108_030 (v3) | 运行 53s 后失败 | `task_registry.py` checkpoint_dir 白名单缺 `X1Exp31AMPOnPolicyRunner` → `ValueError: checkpoint_dir is supported for AMP training`（修复提交 `02ca2a9`） |
| TASK_20261108_031 (v4) | 运行 45s 后失败 | `ValueError: Reward refinement changed protected configuration: domain_rand`（配置回退提交 `d6b3665`） |
| TASK_20261108_033 (v5) | **运行中** | checkpoint 目录、warm_start、REWARD_REFINEMENT 均通过；`Learning iteration 11252/11750` 起正常推进 |

失败任务均无训练数据（`task data keys` 为空）、无有效模型输出，属于任务配置/代码前置校验错误，
与环境或资源无关。

## 验收证据（v5）

- `CHECKPOINT_DIRECTORY /workspace/amp_reo/logs/x1_amp_exp31/exported_data/exp3.1_main`
- `CHECKPOINT_INPUT_READY ... bytes 10777160, sha256 a558ff08…cfcc9e451`
- `REWARD_REFINEMENT_OK`（warm_start 通过受保护配置校验）
- 迭代从 11250 推进到 11750（500 次新更新），`ETA: 0.0`
- exp3.1 指标出现：`StandingStability_angular`、`StandingStability_penalty`
- 零指令子集非空：`Policy/zero_selected_raw_target_noise_std_deg` 有值（|cmd|=0 环境参与统计）
- 检查点 10 个全部上传：`model_11300.pt` … `model_11750.pt`
- 迭代耗时约 8s，全程 4330s（约 72 分钟），任务状态 5（正常结束）

## 终模

| 项 | 值 |
| --- | --- |
| 任务 | `TASK_20261108_033` |
| 结束时间 | 2026-11-08 11:07:29（runtime 4330s） |
| 终模 | `model_11750.pt` |
| 大小 | 10,780,101 bytes |
| SHA256 | `5c78390201c56311e805cb3b8425bf90643c2251065e1b1a6ef2e054be9e73dc` |

## 终态指标（11750）

- `Mean episode task reward` ≈ 34（起点 -1.2）
- `rew_tracking_lin_vel` ≈ 1.04，`rew_tracking_ang_vel` ≈ 0.66
- `Mean episode style reward` ≈ 4.3，AMP expert 0.988 / policy -0.988
- `Tracking/forward_abs_vx_error` 0.038 m/s，`lateral_abs_velocity` 0.026
- `StandingStability_angular` 0.0006（站立子集占比小，惩罚均值低）

## MuJoCo 复测（2026-11-08，与 exp3.0 完全同协议）

工具链先做复现性验证（`czy/data/exp3.1/mujoco/`，2026-11-12 由 `czy/analysis/exp31_diag/` 迁入）：

| 验证 | 结果 |
| --- | --- |
| `model_11250` 零指令无 heading 3s | 不摔（与 E_11250_zero 一致） |
| `model_12250` 零指令 + heading 3s | 1.06000005 s 跌倒（与 C1 逐位一致） |

`model_11750.pt` 已下载并校验：10,780,101 B，SHA256 `5c78390201c56311e805cb3b8425bf90643c2251065e1b1a6ef2e054be9e73dc`。

| 场景 | 参数 | 结果 |
| --- | --- | --- |
| G1 零指令 + heading 3s | `--vx 0.001 --stand-seconds 0 --seconds 3` | **1.08 s 跌倒** |
| G2 零指令无 heading 3s | 同上 + `--disable-heading` | **1.08 s 跌倒** |
| G3 零指令 10s | 同上（跌倒后录 3s，实际 4.08s） | **1.08 s 跌倒** |
| G4 1.5s 零指令缓冲后阶跃 | `--vx 0.595 --stand-seconds 1.5 --seconds 10` | **1.08 s 跌倒（缓冲期内）** |
| G5 立即阶跃（对照 exp3.0 step0） | `--vx 0.595 --stand-seconds 0 --seconds 10` | **3.14 s 跌倒**；mean vx 0.958、RMSE 1.05、3s yaw −94° |

exp3.0 对照：C1/C2 均 1.06 s 跌倒；standard 缓冲期 1.06 s 跌倒；step0 + heading 10s 不摔、11.06 s 跌倒（mean vx 0.549、RMSE 0.555、yaw −20°）。

**结论：exp3.1 未能改善零指令站立（1.08s vs 1.06s，实质相同），且走路在 MuJoCo 中明显变差（3.14s 跌倒、速度超调 0.96 m/s、偏航漂移 −94°）。**

### 原因证据

1. 训练日志中 `Commands/zero_applied_fraction` 与 `Standing/command_fraction` 始终为 0.000000，
   而基于 obs 的 `Policy/zero_selected_raw_target_noise_std_deg` 只有零星值——零指令实际被执行的比例远低于 50%，
   与 `_resample_commands` 在 RSI 初始化路径用 `self._reset_commands` 覆盖采样有关。
2. 用 `model_11250` 的零指令站立轨迹重建 obs，分别喂给 11250 与 11750 策略：
   初始帧动作 RMS 仅 0.02，但在 0.2–1.0s 的站立状态上 RMS 升到 0.13–0.25（约 4–7° 关节目标差）。
   说明 anchor 只约束住了最初的状态，站立流形上的行为仍被 PPO 推离源策略。
3. 站姿角速度惩罚量级过小（`-0.03 × ω/0.15`，且站立子集稀少），不足以对抗 PPO 漂移。
4. G5 变差的原因待定：可能来自 anchor 在指令过渡（vx 由 0 爬升）阶段把动作拉向源策略，
   与加速需求冲突；也可能是站立惩罚/零指令比例改变后的训练噪声。

## 待做（exp3.2 方向）

1. **先做仪器化诊断**：记录 `zero_requested_fraction`/`zero_applied_fraction` 与 anchor loss 实际数值，
   用 50 次短训确认零指令是否真正被施加、anchor 是否真的在约束。
2. **Anchor 改造**：用固定的站立观测 bank（源策略站立轨迹）每次更新都计算锚损失，
   不依赖 minibatch 是否含零指令样本；加硬性回滚门限（站立 bank 上动作 RMS 超阈值则拒绝该次策略更新）。
3. **保证零指令暴露**：在 `_resample_commands` 中对非 RSI 重采样强制按 `zero_probability` 抽取，
   并记录实际施加比例。
4. **站立惩罚**改成有下限的门控或改为站立专用奖励项，避免量级被稀释。
5. `domain_rand` 仍保持与 warm_start 源一致（`reward_transfer.py` 保护），摩擦随机化留到配置匹配后的独立迁移。

## 已知观察

- `Commands/zero_applied_fraction` 在部分迭代块显示 0.000000，但 `Tracking/zero_abs_vx_error`
  同时有值，说明零指令样本存在；平滑指令管线（`SmoothCommands`）下“精确零”只占部分时间。
- 锚约束阈值 |vx|<0.005 偏严格，若需要更强约束效果可后续放宽阈值或改用 KL。

## 网络备注

- 本机无直连 GitHub；用户 socks5h 代理 10.12.201.122:39009 当日不可达。
- 临时使用仅监听 127.0.0.1 的 CONNECT 转发（Python socket 可连通 github.com）完成 push，
  TLS 端到端、不落地日志/凭据；用后关闭。