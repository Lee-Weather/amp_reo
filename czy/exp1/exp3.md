# exp3 系列实验记录

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp3.0 | 2026-11-06 | 延续 MLP 11250；平地、阶跃指令、1.5 s 站姿缓冲；vy 航向观测 + wz 航向命令；训练完成并回放三件套 | 训练完成，待完整验收（A/B 通过，C 待 MuJoCo） | TASK_20261106_088（输入 087） | limxbe2b0a9iij6r@fldubai.com | model_12250.pt |
| exp3.1 | 2026-11-08 | KL 锚点 + 站姿角速度惩罚、零指令比例 50%；MuJoCo 零指令 1.08 s 跌倒、立即阶跃 3.14 s 跌倒 | 失败 | TASK_20261108_033 | limxbe2b0a9iij6r（userId 4741） | model_11750.pt（SHA 5c78…e73dc） |
| exp3.2 | 2026-11-09 | 固定站立锚点库（256 帧）+ 双锚损失 + 零指令重采样修复；MuJoCo 五场景 1.74–3.12 s 全部跌倒 | 失败 | 未留档（taskName `exp3.2-fix-zero-command-v1`） | 未留档 | model_11750.pt（SHA 391f…3bafc） |
| exp3.3 | 2026-11-12 | 用部署 pd_stand 站姿替换零指令站姿；训练 200 轮完成；复跑显示站姿每 1.4 s 跌倒、指令从未生效 | 失败（两处根因，见 §7） | TASK_20261111_106 | limxbe2d78owe5wc（userId 4742） | model_11950.pt（SHA 13ce…af3f） |

当前实验为 **exp3.3**，源模型为 exp3.2 的 `model_11750.pt`（SHA `391f8346…1413bafc`），下一轮为 **exp3.4**。exp3 系列仍未通过 MuJoCo 零指令站立；3.3 复跑证明两个独立缺陷：站姿四元数顺序错误（已修）和零指令/站姿保持从未被真正训练（`Commands/zero_applied_fraction` 在 3.1、3.2、3.3 三代训练里都≈0）。

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

MuJoCo 复测（2026-11-08，与 exp3.0 同协议；证据 `czy/exp1/records/exp3.1_evaluation/mujoco/`）：

| 场景 | 参数 | 结果 |
| --- | --- | --- |
| G1 零指令 + heading 3 s | `--vx 0.001 --stand-seconds 0 --seconds 3` | **1.08 s 跌倒** |
| G2 零指令无 heading 3 s | 同上 + `--disable-heading` | **1.08 s 跌倒** |
| G3 零指令 10 s | 同上（跌倒后录 3 s） | **1.08 s 跌倒** |
| G4 1.5 s 缓冲后阶跃 | `--vx 0.595 --stand-seconds 1.5 --seconds 10` | **1.08 s 跌倒（缓冲期内）** |
| G5 立即阶跃 | `--vx 0.595 --stand-seconds 0 --seconds 10` | **3.14 s 跌倒**；mean vx 0.958、RMSE 1.05、3 s yaw −94° |

**结论**：❌ 未达标。零指令站立与 3.0 实质相同（1.08 s vs 1.06 s），行走反而更差。

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

MuJoCo 复测（2026-11-09，与 3.1 同协议；证据 `czy/exp1/records/exp3.2_evaluation/mujoco/`）：

| 场景 | 结果 | 对比 3.1 |
| --- | --- | --- |
| G1 零指令 + heading 3 s | **1.74 s 跌倒** | 1.08 s |
| G2 零指令无 heading 3 s | **2.22 s 跌倒** | 1.08 s |
| G3 零指令 10 s | **2.22 s 跌倒** | 1.08 s |
| G4 1.5 s 缓冲后阶跃 10 s | **1.76 s 跌倒（缓冲期内）** | 1.08 s |
| G5 立即阶跃 10 s | **3.12 s 跌倒** | 3.14 s |

**结论**：❌ 未达标。站立时间从 ~1.1 s 提升到 1.7–2.2 s，但仍远低于 10 s；立即阶跃无改善。

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

**根因 A：站姿四元数顺序写错。** `x1_amp_exp33_env.py` 把 `standing_state` 的四元数写成 w-first `[1,0,0,0]`，而环境与所有轨迹一致按 **xyzw** 解析（`x1_amp_env.py` 里 `qx,qy,qz,qw = state[3:7]`、零偏航四元数是 `[0,0,0,1]`），于是站姿被读成绕 x 轴 180°，机器人倒置出生。CSV 证据：base_euler_x 179.4°、base_height −0.528 m、足底接触力 0；checkpoint 的 `control.standing_state[3:7]` 同样记录为 `[1,0,0,0]`，说明训练期的站姿重置也是倒置的，训练中这批样本一出生就终止。

修复：`520c76f` 改为 `[0,0,0,1]`。由于站姿属于 checkpoint 的 control metadata，改后 `runner.load` 会拒载；`e5a0dfa` 增加 `--relaxed_metadata`（仍校验关节名、观测维度和历史长度），专门用于"有意改站姿"的回放。

**复跑（2026-11-12，服务器 `exp_20261112_135836`，代码 `520c76f`+`e5a0dfa`，部署 165 个文件哈希全部匹配，模型 SHA 与训练一致，seed 5，3000 步）：**

| 模式 | 终止 | 平均回合 | 终止时 base_height | command_applied | mean vx | yaw 漂移 |
| --- | --- | --- | --- | --- | --- | --- |
| 站姿 | 42/3000 | 71.4 步（1.43 s） | 0.201 m | 0.000（全程为 0） | −0.637 m/s | 最大 174.7° |
| RSI | 41/3000 | 72.1 步（1.44 s） | 0.205 m | 0.002（个别步到 0.4） | −0.618 m/s | 最大 168.2° |

复跑后机器人不再倒置（base_height 均值 0.566 m、euler_x 均值 −2.9°，能站约 1.4 s），但**每 1.4 s 必跌，恰好卡在 1.5 s 零指令缓冲结束之前**，所以指令始终没机会生效（站姿模式 applied 全程 0；RSI 只有 16/3000 步、约 0.53% 的步数放出指令），一旦放出指令机器人以 −0.6 m/s 倒退走。

**根因 B：零指令行为从未被训练。** 3.1 记录已指出零指令实际施加比例≈0，3.2 的"修复"后仍只有 0.006，3.3 训练为 0.0000；而站姿缓冲要求策略在零指令下保持 1.5 s。策略只在正向速度指令下训练过（native 0.595），零指令下是外推行为；3.3 的倒置站姿又让这批样本全部立即终止，等于没有样本。因此"换站姿"本身解决不了零指令站立。

**结论**：❌ 未达标。3.3 的模型不能作为零指令能力的证据（训练期站姿重置是倒置的）；修复四元数后暴露出的真实缺口是策略在零指令下撑不过 1.5 s，且放出指令后会倒退。

**遗留疑问**：pd_stand 右髋 pitch 写成 −0.40（左 +0.40），而同组膝 0.49、踝 −0.21 左右同号；URDF 里右髋限位 [−2, 1] 与左 [−1, 2] 是镜像的，两种符号约定都解释不全。exp3.4 前需要用 FK 或部署配置确认这组值，避免再次在错误的站姿上训练。

**下一轮方向（exp3.4）**：

1. 用修正后的站姿重训，不沿用 11950（它是在倒置站姿上训出来的）。
2. 先修零指令暴露：`Commands/zero_applied_fraction` 必须真的到达 ~0.5，否则锚和惩罚都作用在空集上；用 50 轮短训验证这个指标再看长训。
3. 加"固定 PD 站立 5 s"门禁（`record_standing.py`）作为进入 PPO 前的静态验收。
4. 回放先只看 `--reset_mode=standing` 10 s 的缓冲期是否不摔，再谈速度跟随。

**证据位置**：

- 基线（倒置站姿）：`czy/exp1/records/exp3.3_evaluation/baseline/`（`standing_*`、`rsi_*` 的 CSV、metadata、日志）。
- 复跑（修正后）：`czy/exp1/records/exp3.3_evaluation/rerun_fixed/`（CSV、metadata、日志，以及两段 60 s / 1920×1080 / 25 fps 视频）。
- 训练曲线与任务：`TASK_20261111_106`（flux 可按 key 拉 11751–11950）。
- 站姿修复与回放开关注释见 `520c76f`、`e5a0dfa`。
