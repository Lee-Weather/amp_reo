# exp3 系列实验记录

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp3.0 | 2026-11-06 | 延续 MLP 11250；平地、阶跃指令、1.5 s 站姿缓冲；vy 航向观测 + wz 航向命令；训练完成并回放三件套 | 训练完成，待完整验收（A/B 通过，C 待 MuJoCo） | TASK_20261106_088（输入 087） | limxbe2b0a9iij6r@fldubai.com | model_12250.pt |
| exp3.1 | 2026-11-08 | KL 锚点 + 站姿角速度惩罚、零指令比例 50%；MuJoCo 零指令 1.08 s 跌倒、立即阶跃 3.14 s 跌倒 | 失败 | TASK_20261108_033 | limxbe2b0a9iij6r（userId 4741） | model_11750.pt（SHA 5c78…e73dc） |
| exp3.2 | 2026-11-09 | 固定站立锚点库（256 帧）+ 双锚损失 + 零指令重采样修复；MuJoCo 五场景 1.74–3.12 s 全部跌倒 | 失败 | 未留档（taskName `exp3.2-fix-zero-command-v1`） | 未留档 | model_11750.pt（SHA 391f…3bafc） |
| exp3.3 | 2026-11-12 | 用部署 pd_stand 站姿替换零指令站姿；训练 200 轮完成；复跑显示站姿每 1.4 s 跌倒、指令从未生效 | 失败（两处根因，见 §7） | TASK_20261111_106 | limxbe2d78owe5wc（userId 4742） | model_11950.pt（SHA 13ce…af3f） |
| exp3.4 | 2026-11-13 | 站姿 FK 校验 + 零指令暴露管道修复（退化区间跳过零采样）；从 3.2 的 11750 热启动 500 轮 | 完成：管道修好（zero_applied 0.21–0.30），但站立/行走均回退（12250：33–44 次重置/3000 步，0.4 段倒走 −0.15）；MuJoCo 被 runtime 审计挡住 | TASK_20261113_071 | limx16c492gkh03c（userId 4768） | 失败（负结果，见 §7） |
| exp3.5 | 2026-11-13 | 修 3.4 暴露的"缓冲↔回合长度"死锁：站姿回到运动库帧、`resampling_time` 6→1 s、`stand_buffer_seconds` 1.5→0.4 s；200 轮 | 失败：调度改动生效（中途指令占比 0.50、缓冲 0.11）但零指令仍每 1.4 s 跌倒、行走更差（0.4 段 −0.42）；见 §10 | TASK_20261113_134 | limx16c492gkh03c（userId 4768） | model_11950.pt（SHA 794c…4455） |

当前实验为 **exp3.5（训练中）**，exp3.4 已完结（负结果，§7）。exp3.5 直接针对 3.4 发现的两个机制：`stand_buffer_seconds` 1.5 s（75 步）比零指令回合（71 步）还长，导致行走指令永远发不下去；以及 `resampling_time` 6 s 让短回合里根本不会发生"走到一半被要求停下"。改动只有三处（站姿回运动库帧、指令周期 1 s、缓冲 0.4 s），其余与 3.4 完全一致，便于单变量归因。

**归档约定（2026-11-12 统一）**：每个实验在 `czy/data/{实验编号}/` 保存 `model_{iter}.pt`、`play_output.mp4`、`isaac_diag.csv` 三件套（不区分是否达标）；MuJoCo、RSI、修复前基线等附加回放放该目录的子目录（`mujoco/`、`rsi/`、`baseline/`）；metadata、日志、机读摘要留在 `czy/exp1/records/{实验编号}_evaluation/`。exp3.1 与 3.2 的 Isaac 回放是 2026-11-12 为归档补跑的（之前只有 MuJoCo）。

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


### MuJoCo 回放（2026-11-06）

同一 `model_12250.pt`，平地、校准 runtime、seed=5、无噪声/推力；使用 `scripts/amp_tools/sim2sim_amp_mlp.py`（LF 快照，避免 CRLF 导致 URDF SHA 不符）与 exp3.0 的 heading 配置。结果**未通过**：

| 条件 | 结果 | 视频/CSV |
| --- | --- | --- |
| 标准 1.5 s 零指令缓冲后阶跃 0.595 | **1.06 s 跌倒**（缓冲期内；1.5 s 时 yaw≈177°，剧烈旋转） | [sim2sim_12250.mp4](czy/data/exp3.0/mujoco/sim2sim_12250.mp4)（102 帧/4.06 s） |
| 取消站立缓冲、t=0 立即阶跃 + heading | **10 s 不摔，11.06 s 跌倒**；3.5/10 s 绝对航向约 23.6°/27.4°，3.5–10 s mean vx=0.254、RMSE=0.367 | [sim2sim_12250_step0_10s.mp4](czy/data/exp3.0/mujoco/sim2sim_12250_step0_10s.mp4)（250 帧/10 s）、[sim2sim_12250_step0_60s.mp4](czy/data/exp3.0/mujoco/sim2sim_12250_step0_60s.mp4)（352 帧/14.06 s，含跌倒后 3 s） |
| 立即阶跃、关闭 heading | 3.36 s 跌倒 | 诊断，未保留为主视频 |

**结论：exp3.0 的 Isaac A/B 通过不能转化为 MuJoCo 通过。** 主要失败点是**零指令站立在 MuJoCo 中 1.06 s 内失稳**；取消站立缓冲后能走约 10 s，但速度仅 0.254 m/s、RMSE 0.367、航向偏 23–27°，仍远未达 A。该结果与 exp1.9 MLP 11250 的 10 s 不摔形成对比，说明 exp3.0 的 heading/step 微调可能过拟合 Isaac 的站立/接触条件。

视频均为 1280×720/25fps，全帧解码通过；CSV 数值有限。主视频哈希：standard `10da18e6…4d16`、10 s `16f87015…ed07`、60 s `74a8301e…a2aa`；CSV 哈希见 `czy/exp1/records/exp3.0_evaluation/`。MuJoCo 的视频/CSV 放在 `czy/data/exp3.0/mujoco/`，不改变根目录的三件套。

**下一步**：优先修复 MuJoCo 站立/接触差异（更大接触随机化、MuJoCo-like contact、站立 init/噪声、或单独训练站立保持），再重训；不能以 Isaac 三件套代替 MuJoCo 验收。


## 实验 exp3.1：KL 锚点与零指令站立保持

### 1. 上一实验结果与教训

> 数据：exp3.0 Isaac 三件套（`czy/data/exp3.0/`）与 MuJoCo 复测（`czy/data/exp3.0/mujoco/`）。
> - Isaac A/B：10 s 无跌倒，3.5–10 s mean vx 0.596、RMSE 0.020、3.5→10 s 净航向 8.42°。
> - 同一模型进 MuJoCo：零指令站立 **1.06 s 跌倒**；取消缓冲后 11.06 s 跌倒，mean vx 0.254、RMSE 0.367、航向 23.6°/27.4°。
>
> **核心教训**：
> - Isaac A/B 通过不能迁移到 MuJoCo，站立段最先崩。
> - 零指令在训练里几乎没有样本，策略在零指令下没有参照行为。
> - 本轮要解决：让零指令真正出现（50%），并在零指令时把策略约束回源模型 11250 附近。

### 2. 本轮修改目标

- 零指令比例 20% → 50%（`commands.zero_probability = 0.5`）。
- KL 锚点：`|vx|≈0` 时约束策略输出接近源模型（`anchor_loss_coef = 0.5`，阈值 0.01，obs[:,0]=2·vx）。
- 站姿角速度惩罚：`vx<0.05` 时 `-0.03 × min(ω,0.15)/0.15`。
- 从 `model_11250` 续训 500 轮（11250→11750），保持控制/AMP/奖励口径不变。
- 验收：MuJoCo 零指令 10 s 不摔、缓冲后阶跃 10 s 不摔。

### 3. 修改内容

### 修改一：零指令比例与站姿惩罚

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| `commands.zero_probability` | 0.2 | 0.5 | 每 6 s 重采样，一半给零指令 |
| `rewards.standing_stability_angular_scale` | — | −0.03 | 仅 `|vx|<0.05` 生效 |
| `rewards.standing_stability_angular_max` | — | 0.15 | 归一化上限 |

**理由**：3.0 的站立失败源于零指令样本稀缺，先把暴露量拉起来。

### 修改二：KL 锚点

- `humanoid/amp/x1_exp31_runner.py` 冻结源策略副本为 `policy_anchor`，每个 mini-batch 对 `obs[:,0]<0.01` 的行计算 σ 归一化的动作 MSE（`anchor_loss_coef=0.5`）。

**理由**：不依赖奖励工程，直接把零指令行为拉回已证明能站 10 s 的源模型。

### 修改三：训练方式与域随机

- 训练与源同 `domain_rand`（`reward_transfer.py` 对受保护配置做校验，摩擦随机化被移除，留作配置匹配后的独立迁移）。
- 迁移方式：reward refinement，保留 Actor/D/normalizer/D Adam，重建 Critic/PPO/replay。

### 4. 修改文件

- `humanoid/envs/x1/x1_amp_exp31_config.py`、`x1_amp_exp31_env.py`：新任务、站姿惩罚与指标。
- `humanoid/amp/x1_exp31_runner.py`：锚损失。
- `humanoid/envs/__init__.py`：注册 `x1_amp_exp31`。
- `humanoid/utils/task_registry.py`：runner 解析、`checkpoint_dir` / `warm_start` 白名单。
- 分支 `codex/exp3-0`；训练提交 `d6b3665`（含 `02ca2a9` 白名单修复与 `d6b3665` domain_rand 回退）。

### 5. 训练参数与任务

| 项 | 值 |
| --- | --- |
| 训练方式 | 续训（reward refinement） |
| 源 | exp1.9 `model_11250.pt`，SHA `a558ff08e29c8840665353a5b30f24674562c21b6fdb6c0c47ec0f0cfcc9e451` |
| 轮数 / 保存 | 500（11250→11750）/ 50 |
| 环境 / seed | 4096 / 5 |
| 账号 / 项目 | `limxbe2b0a9iij6r`（userId 4741） / `PRO_20261106_005` |
| 正式任务 | `TASK_20261108_033`，status 5，2026-11-08 09:55:19–11:07:29，runtime 4330 s |
| 失败任务 | 024/029/030/031（SHA 抄错、仓库 URL 错、runner 白名单缺、`domain_rand` 受保护），均无训练数据 |

### 6. 预期与验收

| 层级 | 通过标准 |
| --- | --- |
| A 零指令 10 s | 0 跌倒，10 s 不掉 |
| B 缓冲后阶跃 10 s | 0 跌倒，3.5–10 s mean vx 0.495–0.695、RMSE ≤0.10 |
| C MuJoCo 同协议 | 与 Isaac 结论一致，不能只用 Isaac 代替 |

### 7. 实验结果

终模 `model_11750.pt`：10,780,101 B，SHA256 `5c78390201c56311e805cb3b8425bf90643c2251065e1b1a6ef2e054be9e73dc`。

训练终态（11750）：Mean episode task reward ≈34（起点 −1.2），`rew_tracking_lin_vel`≈1.04、`rew_tracking_ang_vel`≈0.66，style≈4.3（expert 0.988 / policy −0.988），`Tracking/forward_abs_vx_error` 0.038、`lateral_abs_velocity` 0.026。

MuJoCo 复测（2026-11-08，与 exp3.0 同协议；证据 `czy/data/exp3.1/mujoco/`）：

| 场景 | 参数 | 结果 |
| --- | --- | --- |
| G1 零指令 + heading 3 s | `--vx 0.001 --stand-seconds 0 --seconds 3` | **1.08 s 跌倒** |
| G2 零指令无 heading 3 s | 同上 + `--disable-heading` | **1.08 s 跌倒** |
| G3 零指令 10 s | 同上（跌倒后录 3 s） | **1.08 s 跌倒** |
| G4 1.5 s 缓冲后阶跃 | `--vx 0.595 --stand-seconds 1.5 --seconds 10` | **1.08 s 跌倒（缓冲期内）** |
| G5 立即阶跃 | `--vx 0.595 --stand-seconds 0 --seconds 10` | **3.14 s 跌倒**；mean vx 0.958、RMSE 1.05、3 s yaw −94° |

Isaac 回放（2026-11-12 补跑归档，服务器 `exp_20261112_144500_exp31_exp32`，代码 `85bd4ff`，模型 SHA `5c78…e73dc`，seed 5，六段 0/0.4/0.6/0/0.4/0，3000 步）：

| 段 | 指令 | 实际 vx | RMSE |
| --- | --- | --- | --- |
| 0 / 3 / 5（零指令） | 0 | 0.003 / 0.031 / 0.014 | 0.007–0.105 |
| 1 / 4 | 0.40 | 0.386 / 0.326 | 0.044 / 0.054 |
| 2 | 0.60 | 0.488 | 0.075 |

全程只有 2 次终止，都是 10 s 分段正常超时，**没有跌倒**；高度均值 0.624 m，yaw 漂移最大 32.6°。

**结论**：❌ 未达标（MuJoCo 口径）。但 Isaac 里同一模型零指令站得住、0.4/0.6 跟踪误差 0.04–0.075，与 MuJoCo 的 1.08 s 跌倒形成鲜明对比 → 3.1 的缺口主要是 **Isaac→MuJoCo 迁移**，而不是 Isaac 内训练目标没达成。零指令站立与 3.0 在 MuJoCo 上实质相同（1.08 vs 1.06 s），行走反而更差。

**根因分析**：

- `Commands/zero_applied_fraction`、`Standing/command_fraction` 在训练日志里恒为 0.000000；只有基于 obs 的 `Policy/zero_selected_raw_target_noise_std_deg` 零星有值 → **50% 零指令并没有真正施加**，与 `_resample_commands` 在 RSI 路径用 `_reset_commands` 覆盖采样有关。
- 用 11250 的零指令站立轨迹重建 obs，分别喂 11250 与 11750：初始帧动作 RMS 仅 0.02，0.2–1.0 s 升到 0.13–0.25（约 4–7° 关节目标差） → 锚点只约束住了最初的状态，站立流形上的行为仍被 PPO 推离源策略。
- 站姿角速度惩罚量级过小（−0.03×ω/0.15）且站立子集稀少，不足以对抗漂移。
- G5 变差原因未定：可能是锚点在指令爬升阶段把动作拉向源策略、与加速冲突，也可能是训练噪声。

**下一轮方向**（转入 exp3.2）：

- 先做仪器化诊断：记录 `zero_requested_fraction` / `zero_applied_fraction` 与 anchor loss 实际数值，用短训确认零指令是否真的施加。
- 锚点改用固定站立观测库（源策略站立轨迹 256 帧），每次更新都计算，加硬性回滚门限。
- 在 `_resample_commands` 中对非 RSI 重采样强制按 `zero_probability` 抽取，并记录实际施加比例。
- 站立惩罚改门控或改为站立专用奖励项。

## 实验 exp3.2：固定站立锚点库与零指令重采样修复

### 1. 上一实验结果与教训

> 数据：exp3.1 训练日志与 MuJoCo 复测（`czy/exp1/records/exp3.1_evaluation/`）。
> - 零指令站立 1.08 s 跌倒，与 3.0 的 1.06 s 实质相同；立即阶跃 3.14 s 跌倒、mean vx 0.958。
> - `Commands/zero_applied_fraction` 恒为 0，锚损失只在批次含零指令样本时生效。
>
> **核心教训**：
> - 批次级锚损失依赖批次组成；零指令样本≈0 时锚等于没开。
> - 被动的零指令比例不生效，说明重采样路径在覆盖采样结果。
> - 本轮要解决：让锚损失每轮都生效（固定站立库），并真正修复零指令重采样。

### 2. 本轮修改目标

- 预热时让源策略在强制零指令下跑 256 帧，保存观测为固定站立锚点库。
- 每 mini-batch 在库上计算锚损失（`standing_anchor_coef = 0.8`），不受批次组成影响。
- 修复 `_resample_commands`：移除无条件覆盖名义队列指令 `[NATIVE_SPEED,0,0]` 的行。
- 保留批次锚（0.5）与硬拒绝框架（阈值 0.0 = 暂不启用）。
- 从 `model_11250` 重训 500 轮（11250→11750），配置与源一致。

### 3. 修改内容

### 修改一：固定站立锚点库

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| `standing_anchor_bank_size` | — | 256 | 预热时源策略零指令观测 |
| `standing_anchor_coef` | — | 0.8 | 库锚损失权重 |
| `standing_action_rms_threshold` | — | 0.0 | 硬回滚门限，0 = 禁用 |

**理由**：把锚从"批次里恰好有零指令样本"变成"每次更新都有固定站立参照"。

### 修改二：零指令重采样修复

- 覆写 `_resample_commands`，删掉无条件写入 `[NATIVE_SPEED,0,0]` 的分支，按 `zero_probability=0.5` 抽取。
- 新增指标 `zero_requested_raw` / `zero_applied_raw` / `zero_applied_fraction`。

**理由**：3.1 的零指令实际施加比例≈0，必须先把暴露量测出来。

### 4. 修改文件

- `humanoid/envs/x1/x1_amp_exp32_config.py`、`x1_amp_exp32_env.py`。
- `humanoid/amp/x1_exp32_runner.py`（双锚损失 + 库锚）。
- `humanoid/envs/__init__.py`、`humanoid/utils/task_registry.py`。
- 分支 `codex/exp3-0`；训练提交 `e162921`（含 `91e5f48` 重采样修复、`38103c2`/`28d7137`/`9b18efd`/`366dcef`/`e162921` 修正）。

### 5. 训练参数与任务

| 项 | 值 |
| --- | --- |
| 训练方式 | 续训（reward refinement，源 `model_11250.pt`） |
| 轮数 / 保存 | 500（11250→11750）/ 50 |
| 环境 / seed | 4096 / 5 |
| 账号 / 项目 | 未在本地留档 / `PRO_20261109_001` |
| 任务 | 未留档（taskName `exp3.2-fix-zero-command-v1`，提交 `e162921` 的 500 轮任务） |
| 备注 | 当时未把 Task ID 与账号写入本地状态文件，索引表如实标注"未留档" |

### 6. 预期与验收

| 层级 | 通过标准 |
| --- | --- |
| 训练 | `zero_applied_fraction` 明显 >0（目标 ≈0.5 的重置占比） |
| MuJoCo G1/G2/G3 | 零指令 10 s 不摔 |
| MuJoCo G4 | 1.5 s 缓冲后阶跃 10 s 不摔 |
| MuJoCo G5 | 立即阶跃 10 s 不摔，mean vx≈0.595、RMSE ≤0.10 |

### 7. 实验结果

终模 `model_11750.pt`（3.2 重训版）：10,780,293 B，SHA256 `391f83466f26c76c75b0949f68b1771e01b3b4c5a49642527f1f3aea1413bafc`。

训练早期指标：`zero_requested_raw` 0.000000（新采样指令全是 NATIVE_SPEED），`zero_applied_raw` ≈0.007、`zero_applied_fraction` ≈0.006；重采样修复没有把零指令暴露拉起来。

MuJoCo 复测（2026-11-09，与 3.1 同协议；证据 `czy/data/exp3.2/mujoco/`）：

| 场景 | 结果 | 对比 3.1 |
| --- | --- | --- |
| G1 零指令 + heading 3 s | **1.74 s 跌倒** | 1.08 s |
| G2 零指令无 heading 3 s | **2.22 s 跌倒** | 1.08 s |
| G3 零指令 10 s | **2.22 s 跌倒** | 1.08 s |
| G4 1.5 s 缓冲后阶跃 10 s | **1.76 s 跌倒（缓冲期内）** | 1.08 s |
| G5 立即阶跃 10 s | **3.12 s 跌倒** | 3.14 s |

Isaac 回放（2026-11-12 补跑归档，同一次部署 `exp_20261112_144500_exp31_exp32`，代码 `85bd4ff`，模型 SHA `391f…3bafc`，seed 5，3000 步）：

| 段 | 指令 | 实际 vx | RMSE |
| --- | --- | --- | --- |
| 0 / 3 / 5（零指令） | 0 | 0.004 / 0.031 / 0.015 | 0.007–0.082 |
| 1 / 4 | 0.40 | 0.375 / 0.313 | 0.056 / 0.060 |
| 2 | 0.60 | 0.492 | 0.075 |

同样只有 2 次正常超时、无跌倒，高度均值 0.623 m，yaw 漂移最大 32.1°。

**结论**：❌ 未达标（MuJoCo 口径）。站立时间从 ~1.1 s 提升到 1.7–2.2 s，但仍远低于 10 s；立即阶跃无改善。与 3.1 一样，Isaac 内零指令站立正常而 MuJoCo 1.7–2.2 s 就倒，**迁移差距是主要矛盾**，不是 Isaac 训练不足。

**根因分析**：

- 库锚（256 帧）只是单次快照，策略内部状态分布随训练漂移，库的表征能力下降。
- 零指令仍未被真正施加（`zero_applied_fraction`≈0.006），库锚约束的是源策略的站立观测，不是当前策略在零指令下的行为。
- 硬回滚门限默认 0.0（禁用），没有任何防止策略偏离的硬约束。

**下一轮方向**（转入 exp3.3）：

- 站姿本身：3.0–3.2 一直用 `stand_to_walk` 片段的第 11 帧（体前倾）当零指令站姿，与实际部署站姿不一致；改用部署 pd_stand 直立站姿。
- 保留 3.2 的锚与硬拒绝框架，减少改动面，只换站姿。

## 实验 exp3.3：部署直立站姿替换零指令站姿

### 1. 上一实验结果与教训

> 数据：exp3.2 训练日志与 MuJoCo 复测（`czy/exp1/records/exp3.2_evaluation/`）。
> - 五个场景 1.74/2.22/2.22/1.76/3.12 s 全部跌倒，站立时间比 3.1 有提升但仍远低于 10 s。
> - `zero_applied_fraction`≈0.006，库锚约束的是源策略的站立观测，而不是当前策略在零指令下的行为。
>
> **核心教训**：
> - 3.0–3.2 用的"零指令站姿"是 `stand_to_walk` 片段第 11 帧（体前倾），和实际部署直立站姿不一致，一开始就带偏了零指令的目标姿态。
> - 本轮只换站姿，刻意不动锚、奖励与域随机，把变量压到最小。

### 2. 本轮修改目标

- 用部署 pd_stand 直立站姿同时替换 `default_joint_angles` 与零指令 `standing_state`（29 关节值、base z 0.64）。
- 保留 3.2 的批次锚 0.5 / 库锚 0.8 / 库 256 帧 / 硬拒绝阈值 0.0。
- 从 3.2 的 `model_11750` 热启动 200 轮（11750→11950）。
- 验收：Isaac 站姿 1.5 s 缓冲后 10 s 不摔，再 20 s、60 s，最后 MuJoCo 同协议。

### 3. 修改内容

### 修改一：pd_stand 站姿

| 项 | 旧值（3.2 继承） | 新值 |
| --- | --- | --- |
| 零指令站姿来源 | `stand_to_walk` 片段第 11 帧 | 部署 pd_stand：腰 0、肩 0.15/−0.10、肘 0.30、髋 0.40/0.05/−0.31、膝 0.49、踝 −0.21 |
| `default_joint_angles` | 轨迹平均关节角 | 同一组 pd_stand 值 |
| base 高度 | 轨迹帧 | 0.64 m |

**理由**：零指令应该在"部署站立姿态"上保持，而不是在步态片段的前倾帧上。

### 修改二：热启动方式

- `warm_start_mode = "all_weights"`、`require_reward_transfer = False`、`require_policy_transfer = False`，绕开 `reward_transfer.py` 的 SHA/受保护配置校验（本轮站姿即 protected 配置的一部分）。

### 修改三：其余保持

- 零指令比例 50%、native speed 0.5953696966、站姿角速度惩罚 −0.03/0.15、AMP 与奖励口径均与 3.2 相同。

### 4. 修改文件

- `humanoid/envs/x1/x1_amp_exp33_config.py`、`x1_amp_exp33_env.py`（站姿覆盖与判定）。
- `humanoid/amp/x1_exp33_runner.py`。
- `humanoid/envs/__init__.py`、`humanoid/utils/task_registry.py`。
- 训练提交 `fffe274`；后续修复提交 `520c76f`（站姿四元数）、`e5a0dfa`（`--relaxed_metadata` 回放开关）。

### 5. 训练参数与任务

| 项 | 值 |
| --- | --- |
| 训练方式 | 续训（warm_start `all_weights`，优化器重建） |
| 源 | exp3.2 `model_11750.pt`，SHA `391f83466f26c76c75b0949f68b1771e01b3b4c5a49642527f1f3aea1413bafc` |
| 轮数 | 200（11750→11950） |
| 环境 / seed | 4096 / 5 |
| 账号 / 项目 | `limxbe2d78owe5wc`（userId 4742） / `PRO_20261109_001` |
| 正式任务 | `TASK_20261111_106`，status 5，2026-11-11 14:35:39–15:11:51，runtime 2172 s |
| 早期失败 | `TASK_20261110_142`（status 6，runtime 689 s） |

### 6. 预期与验收

| 层级 | 通过标准 |
| --- | --- |
| A 站姿 10 s | 0 跌倒；1.5 s 缓冲结束后 vx 跟随（3.5–10 s mean 0.495–0.695、RMSE ≤0.10） |
| B 20 s | A 全部；10→20 s 净航向 ≤15°、前进 ≥5.5 m |
| C 60 s | 0 重置；每 10 s 每脚 ≥4 次有效离地；左右支撑差 ≤15% |
| D MuJoCo | 同协议复现 A/B，不能用 Isaac 代替 |

### 7. 实验结果

终模 `model_11950.pt`：10,779,333 B，SHA256 `13ce746b025c3f5a5dc0da7bab4326428308c114c7fa079115acdd157bcafa3f`。

训练（11751–11950，flux 曲线）：`Train/mean_reward` −0.13 → 8.29（末 20 轮 7.96，最大 9.65）；`Train/mean_episode_length` 43 → 501（最大 641）；`AMP/done_fraction` 2.51% → 0.20%；`AMP/anchor_loss` 平在 1.28–1.34；**`Commands/zero_applied_fraction` 0.0000（最大 0.0003）**；`Standing/both_feet_contact` 恒为 0。

Isaac 回放（服务器 `exp_20261111_172509`，seed 5，六段 0/0.4/0.6/0/0.4/0，3000 步，模型 SHA 与训练一致）：

| 模式 | 结果 |
| --- | --- |
| 站姿 | **3000/3000 步全部重置**；base_euler_x 179.4°、base_height −0.528 m、足底接触力 0、`command_applied` 恒 0 |
| RSI | 94/3000 终止；mean vx −0.59、RMSE 0.87；yaw 最大 140°；252 步动作饱和 |

> 基线两行的 `mean vx` 同样是倒地位移：站姿行 3000 步全部倒置出生立即重置；RSI 行只有 2504/3000 步高度 >0.5 m，且 `command_applied` 全程为 0、pitch 均值 −13.7°，即全程后仰漂移而非行走。

**根因 A：站姿四元数顺序写错。** `x1_amp_exp33_env.py` 把 `standing_state` 的四元数写成 w-first `[1,0,0,0]`，而环境与所有轨迹一致按 **xyzw** 解析（`x1_amp_env.py` 里 `qx,qy,qz,qw = state[3:7]`、零偏航四元数是 `[0,0,0,1]`），于是站姿被读成绕 x 轴 180°，机器人倒置出生。CSV 证据：base_euler_x 179.4°、base_height −0.528 m、足底接触力 0；checkpoint 的 `control.standing_state[3:7]` 同样记录为 `[1,0,0,0]`，说明训练期的站姿重置用的也是倒置姿态。

修复：`520c76f` 改为 `[0,0,0,1]`。由于站姿属于 checkpoint 的 control metadata，改后 `runner.load` 会拒载；`e5a0dfa` 增加 `--relaxed_metadata`（仍校验关节名、观测维度和历史长度），专门用于"有意改站姿"的回放。

**复跑（2026-11-12，服务器 `exp_20261112_135836`，代码 `520c76f`+`e5a0dfa`，部署 165 个文件哈希全部匹配，模型 SHA 与训练一致，seed 5，3000 步）：**

| 模式 | 终止 | 平均回合 | 终止时 base_height | command_applied | mean vx | yaw 漂移 |
| --- | --- | --- | --- | --- | --- | --- |
| 站姿 | 42/3000 | 71.4 步（1.43 s） | 0.201 m | 0.000（全程为 0） | −0.637 m/s | 最大 174.7° |
| RSI | 41/3000 | 72.1 步（1.44 s） | 0.205 m | 0.002（个别步到 0.4） | −0.618 m/s | 最大 168.2° |

复跑后机器人不再倒置（base_height 均值 0.566 m），但**从出生起就持续后仰**：pitch 在 0.5 s 已 −11.7°、1.0 s −19.5°，`|pitch|≥20°` 的中位时间 1.02 s；底座同期被推着后移（−0.23 → −0.72 m/s，每回合净后移 0.71 m），最终约 1.4 s 摔倒重置，**正好卡在 1.5 s 零指令缓冲结束之前**。所以视频里是"站住 → 后仰 → 摔倒 → 重置"的循环；表里的 mean vx 是倒地过程主导的后移，**不是倒退行走**。指令因此几乎没机会生效（站姿模式 applied 全程 0；RSI 只有 16/3000 步、约 0.53% 放出）。

**根因 B：零指令行为从未被训练。** 3.1 记录已指出零指令实际施加比例≈0，3.2 的"修复"后仍只有 0.006，3.3 训练为 0.0000；而站姿缓冲要求策略在零指令下保持 1.5 s。策略只在正向速度指令下训练过（native 0.595），零指令下是外推行为；加上训练期站姿重置本身就是倒置姿态，真正进入站立分支的样本既少又不可用。因此"换站姿"本身解决不了零指令站立。

**结论**：❌ 未达标。3.3 的模型不能作为零指令能力的证据（训练期站姿重置是倒置的）；修复四元数后暴露出的真实缺口是策略在零指令下既站不住也止不住后仰——出生后持续后仰，约 1.4 s 摔倒，撑不到 1.5 s 缓冲结束。

> 表格里的 `mean vx` 都是"后仰倒地时的底座位移速度"，不是行进速度；判断是否行走应看 `command_applied` 与足底接触相位，本轮 `command_applied` 基本为 0。

**站姿数值溯源（部分澄清）**：pd_stand 的数值与部署参考配置逐项一致（`czy/origin_x1/agibot_x1_train-main/humanoid/envs/x1/x1_dh_stand_config.py:129-140`：左髋 pitch +0.4、右髋 pitch −0.4、双膝 0.49、双踝 −0.21、髋 roll ±0.05、髋 yaw ∓0.31），说明 3.3 不是抄错，右髋负号是部署约定。但那份参考对应 `resources/robots/x1/urdf/x1.urdf`（12 自由度 DH），本项目用 `X1_29DOF_physically_mirrored.urdf`，两套 URDF 的左右髋 pitch 轴都写 `(0,0,1)`——数值能否直接搬到 29 自由度模型必须用 FK 验证（列为 exp3.4 的 P0）。

**下一轮方向（exp3.4）**：

1. 用修正后的站姿重训，不沿用 11950（它是在倒置站姿上训出来的）。
2. 先修零指令暴露：`Commands/zero_applied_fraction` 必须真的到达 ~0.5，否则锚和惩罚都作用在空集上；用 50 轮短训验证这个指标再看长训。
3. 加"固定 PD 站立 5 s"门禁（`record_standing.py`）作为进入 PPO 前的静态验收。
4. 回放先只看 `--reset_mode=standing` 10 s 的缓冲期是否不摔，再谈速度跟随。

**证据位置**：

- 三件套（复跑 standing，修正站姿）：`czy/data/exp3.3/` 的 `model_11950.pt` + `play_output.mp4` + `isaac_diag.csv`。
- RSI 复跑：`czy/data/exp3.3/rsi/`（`play_output.mp4` + `isaac_diag.csv`）。
- 基线（倒置站姿）：`czy/data/exp3.3/baseline/`（两套 `*_play_output.mp4` + `*_isaac_diag.csv`）。
- metadata、日志、机读摘要与任务 payload：`czy/exp1/records/exp3.3_evaluation/`。
- 训练曲线与任务：`TASK_20261111_106`（flux 可按 key 拉 11751–11950）。
- 站姿修复与回放开关注释见 `520c76f`、`e5a0dfa`。

## 实验 exp3.4：站姿 FK 校验 + 零指令暴露修复（方案）

> 状态：方案已写，未开训。§7 待训练完成后补充。

### 1. 上一实验结果与教训

> 数据：exp3.3 复跑（`czy/data/exp3.3/`）与训练曲线（TASK_20261111_106，11751–11950）。
> - 站姿四元数修好后不再倒置，但**出生即后仰**：pitch 0.5 s −11.7°、1.0 s −19.5°，1.02 s 就 |pitch|≥20°，1.4 s 摔倒重置；每回合净后移 0.71 m。
> - 全程 `command_applied` 为 0（站姿）/16 步（RSI）；`Commands/zero_applied_fraction` 训练期 0.0000。
> - 3.1/3.2 的 Isaac 补跑反而 60 s 零跌倒、0.4/0.6 跟踪 RMSE 0.044–0.075，与 MuJoCo 1.7–2.2 s 跌倒形成对照。

**核心教训**：

1. 站姿必须先做"可执行性"校验（四元数、脚底同高、左右对称），否则整轮训练白做；3.3 就是 200 轮训在倒置站姿上。
2. 零指令暴露是硬前置指标，不是事后统计：三代训练该指标≈0，说明 50% 零指令从未进入策略，锚/惩罚都在空集上生效。
3. 站姿数值本身不是抄错：与部署参考 `czy/origin_x1/agibot_x1_train-main/humanoid/envs/x1/x1_dh_stand_config.py:129-140` 逐项一致；问题是它出自 12 自由度 DH 的 `x1.urdf`，能否直接搬到 29 自由度 physically mirrored URDF 未验证。
4. MuJoCo 与 Isaac 的差距是独立议题（3.1/3.2 已证），本轮先不碰域随机，只解决"站姿 + 零指令暴露"。

### 2. 本轮修改目标

| 门禁 | 目标（量化） |
| --- | --- |
| P0 站姿 FK | 双脚最低点高度差 ≤5 mm、左右脚前后差 ≤10 mm、CoM 投影距支撑多边形边界 ≥20 mm、最低点离地 2 mm |
| P1 零指令暴露 | 20 轮冒烟内 `Commands/zero_requested_fraction` ≥0.45、`zero_applied_fraction` ≥0.30 |
| P2 静态门禁 | ~~固定 PD、seed 5，站立 5 s 0 跌倒~~ → 改为**策略站立验收** | 训练后 `--reset_mode=standing` 10 s 0 跌倒 |
| P3 Isaac | A：零指令 10 s 0 跌倒且 |pitch|≤10°；B：20 s，0.4/0.6 跟踪 RMSE ≤0.10、航向 ≤15°；C：60 s 0 重置 |
| P4 MuJoCo | G1/G2/G3 零指令 10 s 不摔；G4 缓冲后阶跃 10 s 不摔；G5 立即阶跃 10 s，mean vx 0.5–0.7、RMSE ≤0.10 |

非目标：不改 AMP/D 语义与 PD 增益、不新增奖励项、不做 heading 改造、不在本轮追 60 s MuJoCo。

### 3. 修改内容

### 修改一：站姿 FK 校验（P0，先于任何训练）

- 新工具 `scripts/amp_tools/check_pose_fk.py`：纯 `numpy` + `lxml`，不需要 Isaac/torch，可直接在本地跑。
- 输入：`resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf` + 待检姿态（exp34 config 的 `default_joint_angles`、部署原值、3.2 的轨迹站姿帧三选一）。
- 输出 JSON：两只脚各 4 个足底角点的世界坐标、脚底高度差、左右脚前后差、CoM 投影与支撑多边形余量；`--variant right-hip-pitch=±0.4` 两种约定各算一次。
- 判定：四项阈值全过 → 站姿照用；任一项不过 → 按 FK 结果修正，并同步 config 的 `default_joint_angles` 与 env 的 `_PD_STAND_ANGLES`。
- 产出记录在 `czy/exp1/records/exp3.4_evaluation/pose_check.json`。

### 修改二：站姿与采样回归单测（`tests/test_exp34.py`，纯 CPU）

- `standing_state[3:7] == [0,0,0,1]` 且四元数范数为 1（防再次写错顺序）。
- config 的 `default_joint_angles` 与 env 的 `_PD_STAND_ANGLES` 逐项一致（防两处漂移）。
- FK 对称性容差（复用 `check_pose_fk` 的纯函数）。
- `_draw_commands(10000)` 的零比例落在 [0.45, 0.55]（stub env，不需要仿真）。

### 修改三：零指令暴露管道（P1，本轮核心）

新任务 `x1_amp_exp34`：`x1_amp_exp34_config.py` / `x1_amp_exp34_env.py`，继承 exp33（锚、库、硬拒绝、站姿惩罚都不动），新增 `X1Exp34AMPOnPolicyRunner` 只负责指标。

**新增指标**（写进 `extras["training_metrics"]`，自动进 TensorBoard）：

| 指标 | 含义 |
| --- | --- |
| `Commands/zero_requested_fraction` | 重采样后 `requested[:,0]==0` 的比例 |
| `Commands/zero_applied_fraction` | buffer 外、非 transition 的 steady 状态里 `applied==0` 的比例 |
| `Commands/stand_buffer_active_fraction` | `standing_commands.remaining>0` 的比例 |
| `Commands/command_from_reference_fraction` | 重置时命令直接取自参考片段的比例 |
| `Commands/nominal_override_fraction` | 被 nominal/native 覆盖的比例 |

**假设（短训验证）**：`x1_amp_env.py:820` 在 RSI 重置时把 `_reset_commands` 设为参考片段速度；`_resample_commands`（`x1_amp_env.py:862-871`）据此跳过 `_draw_commands`，并把 nominal cohort 强制写成 `[native_speed,0,0]` → 50% 零样本进不了 `standing_commands.requested`，于是 `applied` 永远非零。

**修法**（按诊断结果组合）：

1. 重采样一律重新抽样（走 `_draw_commands`），参考片段速度只用于挑匹配的帧，不再作为命令来源。
2. 抽到零指令的 env 在重置时走 standing 分支：姿态用 `standing_state`、`buffer=1.5 s`、参考帧取站立帧。
3. 去掉 nominal→native 的强制覆盖；若担心早期行走样本不足，前 100 轮保留 25% 覆盖作课程。
4. `StandingCommands.applied()` 语义不变，但指标只在 buffer 外统计，避免把"缓冲期 0"误记成零指令。

**门禁**：20 轮 × 64 env 冒烟必须同时满足 `zero_requested≥0.45`、`zero_applied≥0.30`；否则停在冒烟，不进正式训练。

### 修改四：课程（保守）

- 前 100 轮 `zero_probability=0.3`，之后 0.5。
- 其余奖励/AMP/控制与 3.2/3.3 完全一致；若 P2 通过但策略仍后仰，姿态/重心项留到 exp3.5。

### 4. 修改文件

- 新增 `scripts/amp_tools/check_pose_fk.py`、`tests/test_exp34.py`。
- 新增 `humanoid/envs/x1/x1_amp_exp34_config.py`、`x1_amp_exp34_env.py`、`humanoid/amp/x1_exp34_runner.py`。
- `humanoid/envs/__init__.py`、`humanoid/utils/task_registry.py`（注册与白名单）。
- 若 FK 证伪部署值：`x1_amp_exp34_config.py` 的 `default_joint_angles` 与 `x1_amp_exp34_env.py` 的 `_PD_STAND_ANGLES` 同步修正。
- 分支 `codex/exp3-4`。

### 5. 训练参数与任务

| 项 | 值 |
| --- | --- |
| 任务 | `x1_amp_exp34` |
| 源模型 | `czy/data/exp3.2/model_11750.pt`，SHA `391f8346…1413bafc`（不用 11950：它在倒置站姿上训出来的） |
| 迁移 | warm_start `all_weights` + 优化器重建（同 3.3 路径） |
| 轮数 / 保存 | 500 / 50 |
| 环境 / seed | 4096 / 5 |
| rollout / PPO | 96 / 5 epochs × 16 mini-batch |
| 学习率 | 1e-5 起，KL 自适应上限 1e-4 |
| 账号 / 项目 | `limxbe2d78owe5wc`（userId 4742） / `PRO_20261109_001` |
| 镜像 | `BJX00000001` / `V000412`（Isaac Gym Preview 4） |
| 启动命令 | `gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp_exp34 --headless --num_envs=4096 --seed=5 --max_iterations=500 --run_name=exp3.4_main --warm_start=/workspace/amp_reo/humanoid/exported_data/input_model_11750.pt --warm_start_sha256=391f8346…3bafc` |
| 预计耗时 | 3.3 实测 2172 s / 200 轮 ≈ 10.9 s/轮 → 500 轮 ≈ 1.5 h（不含排队） |

### 6. 预期与验收

| 门禁 | 指标 | 通过线 | 停止条件 |
| --- | --- | --- | --- |
| P0 | 脚底 Δz / 前后差 / CoM 余量 | ≤5 mm / ≤10 mm / ≥20 mm | 不对称且无法解释 → 改站姿或改映射，不进训练 |
| P1 | zero_requested / zero_applied（20 轮，p=0.5） | ≥0.35 / ≥0.30 | <0.1 → 停，先修管道 |
| P2 | ~~固定 PD 5 s~~ 改为策略站立验收（摇椅脚 + 固定 PD 无闭环，见 §7） | 训练后 standing 10 s 0 跌倒 |
| P3-A | Isaac 零指令 10 s | 0 跌倒，|pitch|≤10° | 跌倒 → 停 |
| P3-B | Isaac 20 s 0.4/0.6 | RMSE ≤0.10、航向 ≤15° | |
| P3-C | Isaac 60 s | 0 重置 | |
| P4 | MuJoCo G1–G5 | 零指令 ≥10 s、阶跃 ≥10 s、mean vx 0.5–0.7、RMSE ≤0.10 | 任一失败 → 记录，转 exp3.5 处理接触/摩擦 |

### 7. 实验结果

**P0 站姿 FK 校验：通过。** 新工具 `scripts/amp_tools/check_pose_fk.py`（纯 numpy/lxml，本地可跑）在 `X1_29DOF_physically_mirrored.urdf` 上算出的结果：

| 姿态 | 脚底高度差 | 左右前后差 | CoM 余量 | 结论 |
| --- | --- | --- | --- | --- |
| pd_stand（部署值，右髋 −0.4） | 0.0 mm | 0.0 mm | **+99.2 mm** | pass |
| 同上但右髋 +0.4 | 31.6 mm | 207.0 mm | +13.3 mm | fail（镜像约定错） |
| 轨迹平均姿态（mean29） | 0.8 mm | 4.8 mm | −31.8 mm | fail（行走均值不是静态站姿） |

同时得到两个可用于仿真的数字：脚底最低点距 base_link **0.611 m**（原 `standing_state` 写 0.64 m，出生时悬空 29 mm，已改为 0.611），足底接触面约 200×116 mm。

**P1 零指令暴露：根因找到并修复。** 诊断冒烟（20 轮 × 64 env）打印出的关键事实：

- `Commands/zero_requested_fraction` 恒为 0；`zero_requested_raw` 也是 0；调试输出显示 `command_ranges['lin_vel_x'] = [0.595, 0.595]`（退化）。
- 根因在 `humanoid/amp/stationary.py:17`：`sample_commands` 在 `lin_vel_x` 上下界相等时**故意跳过零采样**（"固定评估指令不得被随机置零"）。而 exp3.0–3.3 的任务都把该区间钉死在 native speed，于是 `zero_probability = 0.5` 从 3.1 起就是死代码，`_draw_commands` 永远只返回 native。
- 修法：`x1_amp_exp34_env.py` 在 `env.zero_exposure_fix` 下自行抽样（native/零各按 `zero_curriculum`），同时覆盖 `reset_idx` 与周期性重采样两条路径。

冒烟验证（20 轮 × 64 env，warm start 11750）：

| 指标 | 修复前（fix=False） | 修复后 p=0.3 | 修复后 p=0.5 |
| --- | --- | --- | --- |
| `Commands/zero_requested_fraction` | 0.000 | 0.210 | **0.370** |
| `Commands/zero_applied_fraction`（buffer 外） | 0.000 | 0.163 | **0.299** |
| `Commands/stand_buffer_active_fraction` | 0.38–0.43 | 0.49 | 0.41 |

零指令通路打通（目标 0.30 已达），顺带发现 runner 不会给该任务设置 `_iteration`（exp2.x 的基类不在 MRO 里），已在 env 补 `set_training_iteration`。

**P2 固定 PD 站立门禁：失败，但它不是资产缺陷。** `record_standing.py --task=x1_amp_exp34 --seconds=5 --seeds 5`：3 个 trial 分别在 **1.66 / 1.70 / 1.76 s** 跌倒，终止时高度 0.07 m、roll ±100–180°、非足部接触力 675–950 N。逐帧看（`standing_diag.csv`）：出生 0.626 m，pitch 从 0° 单调涨到 1.5 s 的 47°，随后前倾摔倒——固定 PD 没有闭环，任何小的重心偏差都会一直积累，所以它不是"站姿不可用"的证据。

佐证：exp3.2 的策略在 Isaac 里零指令能连续站 60 s（3.1/3.2 补跑回放，0 跌倒，见各自 §7）。脚型方面 URDF/MJCF 的踝壳是圆底（最低 8 mm 内接触带仅约 9×30 mm），真实 X1 是平底足，这点仍值得记录，但**不影响策略学会主动站立**。

**P2 处置（已确认）**：放弃固定 PD 门禁，改为训练后的策略站立验收（`--reset_mode=standing`，10 s 0 跌倒）。

**训练**：`TASK_20261113_071`（`exp3.4-zero-exposure-v2`，账号 `limx16c492gkh03c` / userId 4768，分支 `codex/exp3-4` 提交 `dd683f3`）。11:12:39 开跑（此前 26 分钟停在草稿），12:44:12 结束，runtime 5493 s，500 轮 `11750 → 12250`，约 11.3 s/轮，1×4090D 24G。

| 轮次 | zero_applied | standing 早停率 | zero_vx | 零指令回合 | 前向回合 | mean_reward |
| --- | --- | --- | --- | --- | --- | --- |
| 11762 | 0.296 | 1.00 | −0.575 | 66.6 | 93.6 | −4.37 |
| 11898 | 0.286 | 1.00 | −0.564 | — | — | −3.87 |
| 11953 | 0.250 | 1.00 | −0.555 | 78.7 | 130.6 | −4.01 |
| 12107 | 0.219 | 1.00 | −0.582 | 86.1 | 156.3 | −3.46 |
| 12226 | 0.205 | 1.00 | −0.575 | 93.4 | 165.8 | −4.39 |

零指令回合从 66 步长到 93 步（1.3→1.9 s），方向对但始终远低于 3 s 的"早停"阈值（`early_failure_rate` 的定义是终止时回合 ≤3 s 且非 timeout），所以 500 轮后 `Init/standing_early_failure_rate` 仍是 1.00。

**Isaac 验收（model_12250，3000 步 / 60 s，seed 5）**：

| 回放 | 重置 | 0.4 段 | 0.6 段 | 零指令段 vx | 说明 |
| --- | --- | --- | --- | --- | --- |
| standing（零指令） | **44** | — | — | −0.55 ~ −0.60 | 每 1.4 s 摔一次，高度最低 0.29 m，\|pitch\| 1.55 rad |
| buffer-rsi（默认 1.5 s 缓冲） | **41** | 实际 applied 0.028 | 0.016 | −0.45 ~ −0.53 | 回合只有 71 步 < 75 步缓冲，**行走指令从未真正下发** |
| walk（`--stand_buffer_seconds=0`） | **33** | −0.15 | −0.29 | −0.45 ~ −0.55 | 关掉缓冲后指令生效，但机器人**倒退**走并频繁跌倒 |

四个实验的 Isaac 对比（同 3000 步、同 seed）：

| 实验 | 模型 | 重置 | 零指令 vx | 0.4 跟踪 |
| --- | --- | --- | --- | --- |
| 3.1 | 11750 | 2 | 0.003–0.031 | 0.386 |
| 3.2 | 11750 | 2 | 0.004–0.031 | 0.376–0.492 |
| 3.3 | 11950 | 42 | −0.63 | 未测（全程零指令） |
| 3.4 | 12250 | 33–44 | −0.55 | **−0.15**（倒退） |

即：3.4 的微调把 3.2 的行走能力也破坏了（训练里前向回合从约 500 步掉到 165 步，与回放一致），站立则仍停在 1.4–1.9 s。

**两个根因**：

1. **零指令通路（已修）**：`humanoid/amp/stationary.py:17` 在 `lin_vel_x` 退化区间时跳过零采样，导致 3.1–3.3 的 `zero_probability` 全是死代码。3.4 自行抽样后，冒烟里 `zero_requested` 0.37、`zero_applied` 0.30，训练里也稳定在 0.21–0.30（后期被 RSI 比例课程稀释）。
2. **缓冲 ↔ 回合长度互相锁死（新发现）**：`stand_buffer_seconds = 1.5 s = 75 步`，而零指令回合只有 71 步——机器人还没等到行走指令就摔了；摔了以后又从"站立 + 零指令"重置，永远拿不到"走"的样本。训练和回放里都可以看到 `command_applied ≈ 0`、41–44 次重置就是这个机制。这解释了为什么加零指令比例反而让行走退化。

**MuJoCo 验收：被审计文件挡住（未通过，不可放行）**。`sim2sim_amp_mlp.py` 的 `align_runtime` 要求 `runtime.json.asset_urdf_sha256` 与当前资产 URDF 一致；现存唯一审计（exp2.9，2026-11-04）记录的是 `6ab9…`，而仓库里的 `X1_29DOF_physically_mirrored.urdf` 是 `f45f…`（两份本地拷贝字节一致）。审计生成器不在 clean_loop/agibot 两棵树里，无法合法重建，因此不放行。顺带修掉 `prepare_mjcf` 的三个真 bug（重复 `autolimits` 属性、绝对 Windows STL 路径、重复追加 actuator 块）：修完后 MJCF 在 mujoco 3.14 上能正常加载（29 个执行器），只剩审计门禁这一关。

**exp3.5 建议**（基于上面两条根因，不要再加零指令比例）：

1. `resampling_time` 6 s → 1.2 s，让零指令在 1–2 s 的短回合内也能被抽到；
2. `stand_buffer_seconds` 与典型回合长度解耦（0.4 s 或直接去掉），避免"永远处于缓冲"；
3. 站立重置姿态回退到 3.2 的运动库帧（AMP 判别器分布内），`pd_stand` 只作为 FK 校验通过的回放变体；
4. 只跑 200 轮，每 50 轮看 `Episode/zero_completed_length` 是否越过 150 步再继续。

**证据**：`czy/data/exp3.4/`（三件套：`model_12250.pt`、`play_output.mp4`＋`isaac_diag.csv` 为步行回放；`standing/`、`buffer_rsi/`、`walk/` 分场景副目录）；`czy/exp1/records/exp3.4_evaluation/`（`pose_check.json`、`p2_standing/`、`smoke/`、`replay/` 三份 `isaac_*_summary.json` 与日志、`create_task.json`）；服务器目录 `/home/robot/czy/exp1/exp_20261113_101027_exp34`（含 `mj_*.log` 的 MuJoCo 阻塞证据）。

### 8. 执行顺序

> **云端任务状态码**：`flux task create` 只生成**草稿**（`taskStatus=0`），必须再执行 `flux task run --task-id TASK_xxx` 才会真正提交并调度（提交后 2=排队/启动中、3=运行、5=成功、6=失败）。exp3.4 第一次就是停在 0，白等 26 分钟。

1. FK 校验（本地，~15 min）→ 写 `pose_check.json`。✅
2. 站姿结论 + 单测（~30 min）。✅ `tests/test_exp34.py`（服务器 9 passed）
3. 任务/指标/管道修改 + 20 轮冒烟（~1 h）。✅ zero_requested 0.37 / zero_applied 0.30
4. 正式 500 轮（~1.5 h）：`TASK_20261113_071`（排队中）。
5. 取模 + Isaac 回放（服务器，**模型来自 exp3.4，metadata 天然匹配，不需要 `--relaxed_metadata`**）：
   ```
   scp 模型 → /home/robot/czy/exp1/<exp34 目录>/checkpoints/
   xvfb-run -a python -u humanoid/scripts/play_amp.py --task=x1_amp_exp34 --headless --num_envs=1 \
     --checkpoint_path=checkpoints/model_XXXXX.pt --seed=5 --reset_mode=standing \
     --output_dir=logs/replay/standing_seed5
   …同上 --reset_mode=rsi --output_dir=logs/replay/rsi_seed5
   ```
6. MuJoCo G1–G5（`sim2sim_amp_mlp.py`，~30 min）：
   `--root <repo> --checkpoint czy/data/exp3.4/model_XXXXX.pt --mjcf-source czy/data/exp3.1/mujoco/G1_zero_heading/mjcf/x1_29dof_scene.xml --meshes resources/robots/x1_29/meshes --training-commit <task 的 resolved commit> --runtime-properties E:/X1/amp_re0/agibot_x1_train-main/czy/exp1/records/exp2.9_long_evaluation/runtime_audit/runtime.json --passive-probe …/passive_probe.json`（G1/G2/G3 零指令、G4 缓冲阶跃、G5 立即阶跃）
7. 三件套归档到 `czy/data/exp3.4/`（model + play_output.mp4 + isaac_diag.csv），MuJoCo 进 `mujoco/`，metadata 进 `records/exp3.4_evaluation/`；回填 §7 与索引。
8. 检查点：模型到 11950 时先看 `Commands/zero_applied_fraction` ≥0.30 且 `rew_tracking_lin_vel` 掉幅 <20%，再让其跑满 500 轮。

### 9. 风险与决策点

- **部署值在 29 自由度模型上不对称**：若 FK 显示脚底不齐，优先怀疑 DH→29 自由度的轴映射，而不是简单翻转符号；必要时回到"轨迹平均站姿"降级方案。
- **零样本拉高后行走变差**：用 0.3→0.5 的课程，若 `rew_tracking_lin_vel` 在前 100 轮掉超过 20%，先停在 0.3。
- **metadata 拒载**：站姿变会改 control metadata，训练侧走 `all_weights`（3.3 已有路径），回放侧才用 `--relaxed_metadata`。
- **Isaac 通过但 MuJoCo 仍短命**：说明瓶颈在接触/摩擦建模而非训练目标，exp3.5 应转向域随机（3.0 已列方向），不要继续加零指令样本。

---

## exp3.5：解开"缓冲 ↔ 回合长度"死锁

### 1. 背景

exp3.4 把零指令通路修好（`stationary.py:17` 在退化区间跳过零采样），但策略反而更差。回放把机制钉死了：

- 零指令回合只有 **71 步（1.42 s）**，而 `stand_buffer_seconds = 1.5 s = 75 步`——机器人还没拿到行走指令就摔了，`command_applied` 最大只有 0.028；
- `resampling_time = 6 s = 300 步`，而回合只有 71–165 步，**周期性重采样几乎从不触发**（`Commands/command_from_reference_fraction ≈ 1.0`），所以"走到一半被要求停下"这种关键样本根本不存在；
- 站立重置用的是部署 `pd_stand`（不在 AMP 运动库分布内），判别器把策略往"静止保持"推，而静止保持恰好无法在圆底踝壳上站稳。

三者叠加成死锁：摔 → 从"站立+零指令"重置 → 再次摔，永远拿不到"走一段再停"的经验。

### 2. 改动（只动三处，其余与 exp3.4 完全一致）

| 项 | exp3.4 | exp3.5 | 目的 |
| --- | --- | --- | --- |
| 站立重置姿态 | 部署 `pd_stand`（+ 0.611 m） | 运动库站立帧（3.2 那一帧） | 让站姿回到判别器分布内 |
| `commands.resampling_time` | 6 s（300 步） | **1 s（50 步）** | 指令变化落进 2–4 s 的短回合 |
| `amp.stand_buffer_seconds` | 1.5 s（75 步） | **0.4 s（20 步）** | 缓冲不再吞掉整个回合 |

零采样仍按课程 0.25（前 60 轮）→ 0.35；其余奖励、锚点、硬拒绝、`all_weights` 热启动（3.2 的 `model_11750`）都不变。

实现要点：`X1AMPExp35Env._init_buffers` 直接从 `X1AMPExp32Env` 起调 MRO，跳过 exp3.3/3.4 的 `pd_stand` 覆写；`exp34` 的 `standing_pose_z` 改为可空（`None` 时沿用继承来的高度）。

### 3. 冒烟（20 轮 × 64 env，warm start 11750）

| 指标 | exp3.4 | exp3.5 |
| --- | --- | --- |
| `Commands/command_from_reference_fraction` | 1.00 | **0.502**（一半指令变化发生在回合中途） |
| `Commands/stand_buffer_active_fraction` | 0.41–0.49 | **0.113** |
| `Commands/zero_requested_fraction` | 0.37 | 0.306 |
| `Commands/zero_applied_fraction`（缓冲外） | 0.30 | 0.277（课程仍在 0.25 段） |
| 零指令回合 / 前向回合 | 66 / 94 步 | 83 / 77 步 |

结构与预期一致：指令周期和缓冲都已生效。

### 4. 正式训练

`TASK_20261113_134`（`exp3.5-buffer-fix-v1`），2026-11-13 13:39:11 开跑，200 轮 4096 env，分支 `codex/exp3-4` 提交 `530ae2a`，warm start 3.2 的 `model_11750`。

**中途判据**：到 11900（150 轮）时看三个数——`Episode/zero_completed_length` 是否 ≥150 步（3 s，跨过早停阈值）、`Init/standing_early_failure_rate` 是否 <1.0、`Episode/forward_completed_length` 是否仍 >300 步（行走没被拖垮）。三条都满足才做完整回放验收，否则按 §5 继续调。

### 5. 结果：结构改动生效，站立仍然学不出来

训练 `TASK_20261113_134` 于 14:16:09 结束，runtime 2218 s（200 轮 4096 env，11.1 s/轮），产出 `czy/data/exp3.5/model_11950.pt`（SHA `794c6445…`）。

| 轮次 | zero_vx | 零指令回合 | 前向回合 | standing 早停率 | mean_reward |
| --- | --- | --- | --- | --- | --- |
| 11811 | −0.68 | 81.2 | 79.2 | 1.00 | −8.17 |
| 11870 | −0.684 | 78.6 | 78.6 | 1.00 | −5.53 |
| 11950 | −0.677 | 78.5 | 78.5 | 1.00 | −6.03 |

回放（model_11950，1500 步 / 60 s）：

| 回放 | 重置 | 0.4 段 | 0.6 段 | 零指令段 vx | 结论 |
| --- | --- | --- | --- | --- | --- |
| standing | **42**（28 次/千步） | — | — | −0.50 | 比 3.4（14.7 次/千步）更差 |
| walk（`--stand_buffer_seconds=0`） | **39** | applied 0.34，real **−0.42** | applied 0.53，real **−0.46** | −0.47 ~ −0.50 | 指令生效但整段倒走 |

零指令回合从 3.4 的 71–93 步变成 78 步，没有跨过 3 s 门槛；行走反而更差（3.4 的 −0.15 → 现在的 −0.42）。

**跨实验的共同签名**：3.3（−0.63）、3.4（−0.55）、3.5（−0.68，训练内）在零指令下都是"向后漂移 → 1.4–1.9 s 跌倒"，而 3.2 的原模型在**从行走进入**零指令时能站满 60 s。也就是说：问题不是指令调度（3.5 已经证明调度改动不改变结局），而是**从静止站姿冷启动的平衡技能没有梯度可学**。

### 6. 结论与下一步（不再调零指令比例）

三次"加零指令样本"的实验（3.3 / 3.4 / 3.5）都以下跌收场，且都伴随行走退化。建议换攻击面，按优先级：

1. **改站立重置分布，而不是改比例**：让站立回合初始就带 0.2–0.3 m/s 前向速度和步态相位（"正在走 → 收到零指令 → 刹停"），把"冷启动平衡"换成"制动到站立"——后者才是 3.2 已具备、回放里也确实站得住的能力。
2. **加支撑多边形奖励**：用 `scripts/amp_tools/check_pose_fk.py` 的足底凸包（或仿真里的足接触点）算 CoM 到支撑边界的余量，给零指令段直接的可微梯度；现在只有姿态/角速度惩罚，没有"别越界"的信号，圆底踝壳上就很容易一直向后滑。
3. **`pd_stand` 降级为回放变体**：训练用运动库站姿（3.5 已改回），部署站姿只在验收时用 `--reset_mode=standing` 的显式姿态；若一定要在真实机器人上冷启动站立，那是平底足 + 固定 PD 控制器的事，不该由 AMP 策略承担。
4. **MuJoCo 审计先补上**：没有 `runtime.json` 生成器，3.4/3.5 的物理对齐无法验证，继续训练也只是在 Isaac 里自证。

若按 1+2 重做，建议编号 exp3.6，仍然 200 轮、从 3.2 的 11750 热启动，判据不变（`Episode/zero_completed_length` ≥150 步且 `forward_completed_length` >300 步）。

### 7. 若沿用旧调度仍要试

优先怀疑顺序：① 指令周期还不够短（0.6 s 或只在行走后 1 s 处强制插零）；② 零指令样本权重过高（降到 0.2）；③ 站位仍偏窄（`hip_roll` ±0.05 → ±0.02 或加大 `feet_friction`），把圆底踝壳的支撑面纳入域随机。

**证据**：`czy/data/exp3.5/`（三件套 `model_11950.pt`、`play_output.mp4`+`isaac_diag.csv` 为步行回放，`standing/`、`walk/` 副目录）；`czy/exp1/records/exp3.5_evaluation/replay/`（两份 `isaac_*_summary.json` 与原始日志）；服务器 `exp_20261113_101027_exp34` 下的 `x35_*.log`。
