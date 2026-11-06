# exp3 系列实验记录

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp3.0 | 2026-11-06 | 延续 MLP 11250；平地、阶跃指令、1.5 s 站姿缓冲；vy 航向观测 + wz 航向命令；训练完成并回放三件套 | 训练完成，待完整验收（A/B 通过，C 待 MuJoCo） | TASK_20261106_088（输入 087） | limxbe2b0a9iij6r@fldubai.com | model_12250.pt |

当前实验为 **exp3.0**，源模型为 exp1.9 / model_11250（MLP 5×96→29）。平地、无坡面、无速度缓坡；速度指令在 1.5 s 零指令缓冲后阶跃到 0.5953696966 m/s。

---

## 实验 exp3.0：MLP 11250 平地阶跃行走与航向反馈

### 1. 上一实验结果与教训

> 数据：exp2.13 A/B 诊断；exp1.9/model_11250。
> - MLP 11250 在 MuJoCo 平地、1.5 s 缓冲 + 2 s 斜坡、0.595 指令下 **10 s 不摔**，3.5–10 s mean vx=0.606、RMSE=0.097，前进 4.46 m；但 1.5→3.5 航向 19.86°、3.5→10 25.46°、mean abs wz=0.469；20 s baseline 11.72 s 跌倒。
> - 外部 wz 航向命令 gain 0.5 全程：10 s 不摔，3.5/10 s 绝对航向 2.22°/11.80°、3.5→10 14.03°，但 1.5→3.5 变化 18.67°（纠正缓冲期 -16.45° 漂移），mean abs wz 0.484；20 s 仍 13.24 s 跌倒。
> - 阶跃 baseline：10 s 不摔，1.5→3.5 仅 3.18°，但 10 s 绝对航向 -33.45°、mean vx 0.660/RMSE 0.119。
>
> **核心教训**：
> - 14500 DH 的外部航向控制器无效（基线 6.28 s，gain 0.5 4.88 s，damping 1.0 4.84 s）。
> - MLP 11250 是当前唯一 10 s 不摔的模型；航向、站立缓冲漂移和 20 s 稳定性是三个主要缺口。
> - 阶跃不是起步偏航的根因；2 s 斜坡也不是；缓冲期站姿漂移和稳态偏航才是。
> - 本轮明确只考虑平地、阶跃指令，不再设速度缓坡。

### 2. 本轮修改目标

- 从 MLP 11250 继续训练，保持 5×96→29 轻量结构，不引入 phase/CNN/速度估计器。
- 平地、无坡面；1.5 s 零指令缓冲后阶跃到 +0.5953696966 m/s。
- 用 vy 槽传递 heading error、wz 槽传递 heading correction command；先过 10 s A，再追 20 s B 和 60 s C。
- 保持 PD/50Hz/500Hz/action_scale=0.25/无 LPF、AMP 3×86 与 D 语义不变。

### 3. 修改内容

### 修改一：MLP 继承与热启动

- 新任务 `x1_amp_exp30`，配置继承 exp1.9，环境继承 exp1.8 MLP。
- 从 `model_11250.pt`（SHA `a558ff08e29c8840665353a5b30f24674562c21b6fdb6c0c47ec0f0cfcc9e451`）做 `policy_amp_fresh_critic` 迁移：保留 Actor/D/normalizer/D Adam，重建 Critic/PPO Adam/replay；来源 iter 11250。
- 放开了 `reward_transfer.py` 对 DH 类的硬检查，使 MLP 可以使用同一“保留策略与 AMP、重建价值/PPO”的迁移路径。

### 修改二：平地阶跃指令

| 项 | 设定 |
| --- | --- |
| 地形 | 平地 plane，无坡面/台阶/地形课程 |
| 指令 | 1.5 s 零指令后阶跃到 +0.5953696966；无 smoothstep |
| 事件 | 80% +0.595 / 20% 0；重采样 6 s |
| 反向/低速 | 第一阶段不做 |
| 重置 | 50% 站姿 + 50% RSI（exp1.9 继承） |

### 修改三：航向观测与命令

- 每步计算 `heading_error = wrap(yaw − reset_yaw)`。
- vy 槽写入 `clip(error, ±2.0) × 1.0`；奖励中的 vy 目标仍为 0。
- wz 槽写入 `clip(−0.5 × error, ±0.3)`；缓冲期（`standing_commands.remaining>0`）也生效。
- `step()` 在 `standing_commands.applied()`/`sync_policy_commands` 之后恢复最新帧的 vy/wz；历史帧保留各自时刻的值。
- 奖励 `heading_scale=-0.1`、`heading_tolerance=0.25`，避免像 exp2.13 r1 那样让 heading 主导 task。

### 修改四：训练参数

| 参数 | 值 |
| --- | --- |
| 源 | exp1.9/model_11250.pt |
| 轮数 / 保存 | 1000（11250→12250）/ 100 |
| 环境 / seed | 4096 / 5 |
| rollout / PPO | 96 / 5 epochs × 16 小批 |
| D | 16 次/轮，replay 400k，style 10，task 0.5 |
| 学习率 | 1e-5 起，KL 自适应上限 1e-4 |
| 迁移 | 保留 Actor/D/normalizer/D Adam；Critic/PPO Adam/replay 重建 |
| 代码 | Lee-Weather/amp_reo，codex/exp3-0，commit 80438eaf0219c9f3d618b2adf752108b215e86f0 |
| 账号 / 项目 | limxbe2b0a9iij6r@fldubai.com / PRO_20261106_004 |

### 4. 修改文件

- `humanoid/envs/x1/x1_amp_exp30_config.py`、`x1_amp_exp30_env.py`：新任务、航向观测/命令、平地阶跃指令。
- `humanoid/amp/heading.py`：航向纯函数。
- `humanoid/envs/__init__.py`：注册 `x1_amp_exp30`。
- `humanoid/amp/reward_transfer.py`：允许 MLP 的 reward-refinement 迁移。
- `scripts/amp_tools/private_training_input.py`、`flux_network_bridge.py`、`flux_ops.py`：私有 checkpoint 上传工具。

### 5. 训练参数与任务

| 项 | 值 |
| --- | --- |
| 输入容器 | TASK_20261106_087（正确账号 4741） |
| 预检 | 未在正确账号重跑；代码/迁移已在错误账号 077 做过 2 更新接口验证，不作为本账号证据 |
| 正式 | TASK_20261106_088（1000 更新，训练中） |
| 输入模型 | `input_model_11250.pt`，10,777,160 字节，SHA a558…e451 |
| 输出目录 | `/workspace/amp_reo/logs/x1_amp_exp30/exported_data/exp3.0_main` |
| 首模 | 11300（11250 后最近 100 倍数；待正确账号 088 复现） |

### 6. 预期与验收

固定条件：MuJoCo 校准物理、平地、1.5 s 零指令后阶跃 +0.5953696966、无噪声/推力/LPF/重置、seed=5。

| 层级 | 通过标准 |
| --- | --- |
| A 起步/10 s | 0 跌倒；3.5 s 与 10 s 绝对航向 ≤15°；3.5→10 净变化 ≤15°；3.5 s vx ≥0.35；3.5–10 mean vx 0.495–0.695；RMSE ≤0.10；前进 ≥2.5 m |
| B 20 s | A 全部；0 跌倒；10→20 s 净航向 ≤15°；20 s 速度条件；前进 ≥5.5 m |
| C 60 s | B 全部；0 跌倒/重置；60 s 总航向 ≤30°；每 10 s 每脚 ≥4 次 >1 cm 且 ≥80 ms 离地；P95 左≥20/右≥15 mm；低于 1 cm ≤85%；左右支撑差 ≤15%；视觉复核 |
| D 参考步态 | 相位相关、膝 RMSE、摆动 precision/recall、滑移、腰限位、双髋 yaw、上肢 RMSE；留到 C 后 |

### 7. 实验结果

**预检 TASK_20261106_077（2026-11-06）：通过。** 状态 5；输入 10,777,160 字节、SHA 正确；`REWARD_REFINEMENT_OK` 来源 iter 11250，保留 actor/D/normalizer/D Adam（D Adam step 4000、normalizer count 294,912,000），重建 critic/PPO/replay，初始 lr 1e-5；2 次真实更新（11251/11252），日志无 Traceback。早期指标仅 2 轮，不能作步态结论：`Heading/cost≈0.046`、`Heading/error_rad≈0.019`、`Tracking/forward_vx≈0.546`、`Tracking/moving_abs_yaw≈0.230`。

**账号核对与错误账号任务（2026-11-06）：** 首次创建的输入/预检/正式任务（`TASK_20261106_075/077/079`，项目 `PRO_20261106_004`）实际归属于 **`limx1c5eccw64on4@maxxspace.com`（userId 4677）**，不是用户指定的 `limxbe2b0a9iij6r@fldubai.com`（userId 4741）。已停止 `TASK_20261106_079`；其 `model_11300.pt`（10,779,909 字节，SHA `55e45c43…05fd`）及早期指标 **不作为 exp3.0 结果**。错误账号任务保留为事故记录，不删除、不复用。

**正确账号正式任务 TASK_20261106_088（2026-11-06 启动）：** 项目 `PRO_20261106_005`，账号 userId 4741，commit 80438ea；输入容器 `TASK_20261106_087` 使用同一私有 11250 对象；正式 088 从 11250 重新迁移，输出目录 `exp3.0_correct_main`。当前训练中；首个 checkpoint、`REWARD_REFINEMENT_OK` 和 300/600/1000 sim2sim 待补充。

> 本轮仅平地、阶跃；不设速度缓坡，不做坡面/台阶。


## 最终训练与 201.5 回放（2026-11-06）

### 最终训练

- 正式任务 `TASK_20261106_088` 状态 5，账号 `limxbe2b0a9iij6r@fldubai.com` / userId 4741，项目 `PRO_20261106_005`，commit 80438eaf0219c9f3d618b2adf752108b215e86f0。
- 最终 `model_12250.pt`：10,779,909 字节，SHA256 `58e24c62a73bd12db0fc81b7e2e6e53cbf9c84ff5ab3c0c19aaf8c6db06bf8b6`；平台 model list 已出现 12250。
- 训练 11250→12250，共 1000 新更新；输出目录 `/workspace/amp_reo/logs/x1_amp_exp30/exported_data/exp3.0_correct_main`。
- 错误账号的 079 已停止，其 11300 模型不用于本结果。

### 201.5 Isaac 回放

服务器目录：`/home/robot/czy/exp1/exp_20261106_132717/amp_reo`，F1 / RTX A6000；部署本地 `codex/exp3-0` 回放代码（含 `x1_amp_exp30` 路由，commit c5e6e1b；env/config 与训练 commit 80438ea 相同），模型 `model_12250.pt` SHA 与训练一致。平地、站姿初始化、seed=5、1.5 s 零指令后**阶跃**到 0.5953696966、无速度缓坡、无噪声/推力、3000 步/60 秒，1920×1080/25fps。

| 回放指标 | 值 |
| --- | --- |
| 非超时终止 / 正常 20 s 超时 | **0 / 2** |
| 3.5 s 绝对航向（相对 t=0） | **1.44°** |
| 10 s 绝对航向 | **6.97°** |
| 3.5→10 s 净航向 | **8.42°** |
| 20 s 绝对航向 | 21.49° |
| 10→20 s 净航向 | 14.52° |
| 60 s 绝对航向 | 21.14° |
| 3.5 s vx（3.25–3.5 均值） | **0.571 m/s** |
| 3.5–10 s mean vx / RMSE | **0.596 / 0.020 m/s** |
| 10–20 s mean vx / RMSE | **0.599 / 0.020 m/s** |
| 3.5–10 s mean abs vy / wz | 0.013 m/s / 0.042 rad/s |
| 3.5–10 s 前进位移 | 3.89 m |
| 10–20 s 前进位移 | 5.85 m |
| 左脚 / 右脚接触力 >20N 占比 | 62.4% / 70.0% |

注意：60 秒回放包含 2 次正常 20 秒超时重置；20–60 s 的分段均值含重置/缓冲，不能当作连续 60 秒稳态。Isaac CSV 只有足 link z 与接触力，没有最低碰撞脚底高度，故 C 的脚底 P95/有效离地时长不能由本 CSV 完整判定；需要 MuJoCo 连续 60 秒或离线 FK 复核。视觉自然度也尚未完成。

**A（10 s，Isaac）通过**：无非超时终止；3.5/10 s 绝对航向 1.44°/6.97°、3.5→10 8.42°；3.5 s vx 0.571、3.5–10 0.596/RMSE 0.020、位移 3.89 m。
**B（20 s，Isaac）通过速度/净航向/无跌倒条件**：10→20 s 净航向 14.52°、mean vx 0.599/RMSE 0.020、位移 5.85 m；但 20 s 绝对航向 21.49°，若要求 20 s 绝对 ≤15° 则未过。
**C（连续 60 s）未完成**：Isaac 回放受 20 s episode 上限，2 次正常超时；60 s 绝对航向 21.14°≤30°，但“0 重置”和脚底离地/有效摆动需 MuJoCo 连续 60 s 复核。

### 三件套

`czy/data/exp3.0/` 恰好三个文件：

| 文件 | 字节 | SHA256 |
| --- | --- | --- |
| model_12250.pt | 10,779,909 | 58e24c62a73bd12db0fc81b7e2e6e53cbf9c84ff5ab3c0c19aaf8c6db06bf8b6 |
| play_output.mp4 | 22,044,826 | c2a3709b275443506a53930fc9a0f30374491f094a719d7879e7a7052d865810 |
| isaac_diag.csv | 24,978,123 | d0fe20647ce3a5d27696e6780aa81f72b4f895772a3effe1a488165ab4b8b5c2 |

验证：MP4 1920×1080、25 fps、1500 帧全部解码；CSV 3000 行、全部数值有限；模型 SHA 与训练回执一致。回放日志与 metadata 在 `czy/exp1/records/exp3.0_evaluation/`。

**下一步**：用同一 `model_12250.pt` 在 MuJoCo 平地阶跃下做连续 10/20/60 s 检查（A/B/C），并补脚底最低碰撞点几何与视觉复核；Isaac 三件套已交付。

> 本轮仅平地、阶跃；不设速度缓坡，不做坡面/台阶。
