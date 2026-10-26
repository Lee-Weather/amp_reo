# X1 29 自由度 AMP（Isaac Gym）

新增任务名 `x1_amp`。保留原来的 `x1_dh_stand`；新任务使用 29 关节全身 PD 控制、PPO 和 AMP 动作风格判别器。参考 `czy/roboparty_train` 的 LSGAN 奖励和梯度惩罚方式，在本工程的 Isaac Gym Preview 4 中实现，不需要安装 Isaac Lab。

## 输入与模型约定

- URDF：`resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf`，网格来自其相对路径 `../meshes/`。
- 轨迹：`resources/motions/slow_walk_amp_v1.2/normalized/*.pkl`，默认加载四条片段，按片段等概率采样。
- 每条轨迹必须有 `fps`、`dof_names`、`dof_pos`、`root_pos`、`root_rot`。四元数顺序为 **xyzw**。
- 按仿真器实际关节名重排数据，不能假定 Isaac Gym 的关节顺序等于 MJCF 的顺序。当前 29 个动作覆盖腰 3、双臂 14、双腿 12 个关节。
- 所有片段按真实时间插值；不跨片段、不把末帧连接回首帧。即使片段标注 `periodic`，也不默认首尾已经平滑对齐。
- 判别器关键点通过当前 URDF 正向运动学生成，避免依赖其他模型的缓存关键点。采用双踝、双膝、双肘共 6 个 link 原点。
- 原始片段有少量关节限位越界。默认允许最多 0.05 rad 并警告、裁剪后重新计算速度；超出阈值直接报错。不会修改原始文件或猜测关节正负号。

## 训练

沿用主项目的 Isaac Gym、PyTorch、CUDA 环境。服务器已有的 `F1` 环境可以运行；无需修改共享环境的包安装路径，设置当前工程 `PYTHONPATH` 即可。

Linux / 服务器，在工程根目录执行：

```bash
source /home/robot/Anaconda/etc/profile.d/conda.sh
conda activate F1
export PYTHONPATH="$PWD"
python humanoid/scripts/validate_amp.py --task=x1_amp --num_envs=8 --headless
python humanoid/scripts/train.py --task=x1_amp --num_envs=4096 --run_name=amp29 --headless
```

先进行少量迭代的连通性检查：

```bash
python humanoid/scripts/train.py --task=x1_amp --num_envs=64 --max_iterations=20 --run_name=amp_smoke --headless
```

Flux 云端只配置当前 `amp_reo/main` 仓库。训练需要的 URDF、61个网格及4段规范化轨迹均位于 `resources/`，随代码一并拉取。直接执行训练入口：

```bash
gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp --num_envs=4096 --max_iterations=6000 --run_name=amp29_6000 --headless
```

Flux 应选择 Isaac Gym Preview 4 镜像；`hparamsPath` 为 `amp_reo/humanoid/envs/x1/x1_amp_config.py`。原始 `czy/` 模型和轨迹目录保留在本地并继续被 Git 忽略。

日志和检查点保存在 `logs/x1_amp/exported_data/<时间戳><run_name>/`。包含 TensorBoard、`config.json` 和 `model_<iteration>.pt`。

恢复训练（将 `<run>` 替换为实际目录名）：

```bash
python humanoid/scripts/train.py --task=x1_amp --resume --load_run='<run>' --checkpoint=20 --num_envs=4096 --max_iterations=1000 --run_name=amp29_resume --headless
```

这里 `max_iterations=1000` 表示再训练 1000 次更新。恢复策略、价值网络、判别器、判别器归一化、两个优化器和迭代数；经验回放池会由新 rollout 重新填充。检查点会校验关节名、动作零点、动作比例、观测维度、关键点顺序和策略周期。不兼容原 12 自由度 DH 模型。

## 回放和导出

```bash
python humanoid/scripts/play_amp.py --task=x1_amp --load_run='<run>' --checkpoint=6000 --num_envs=1 --headless
```

默认参考 `E:/X1/AMP/humanoid/scripts/play.py`，依次回放 `0 → 0.4 → 0.6 → 0 → 0.4 → 0 m/s`，每段10秒，共60秒。1920×1080跟随相机每两个控制步录一帧（当前25fps），叠加指令/瞬时/平均速度、横向/垂直/偏航速度、左右足接触力和支撑状态，结束输出分段速度误差与重置统计。

默认输出到 `logs/x1_amp/play_output/<时间戳>/`，包含 `play_output.mp4`、`isaac_diag.csv`、`replay_metadata.json`。可用 `--output_dir` 指定新的目录（已有产物时拒绝覆盖）。归档三件套时只复制模型、MP4和CSV，metadata放 `czy/exp1/artifacts/{实验编号}/`，临时文件用完清理，`czy/analysis/`不堆积部署包。`--no_video` 或 `PLAY_RENDER=0` 只关闭录像，CSV和分段汇总仍生成。CSV兼容 `base_vel_*`、`command_*`、`foot_force_l/r` 和逐关节 `action/pos/vel/effort/pos_des_raw` 字段；不填造AMP中不存在的步态相位。终止步使用重置前快照，力矩饱和统计覆盖每个物理子步。

回放重置后默认先给1.5秒零速度缓冲，CSV分别记录requested/applied/observed指令；`--stand_buffer_seconds=0` 可关闭。策略最新观测帧在推理前同步指令，旧历史帧保持原样。这个缓冲参考提供的play脚本，并不代表本AMP训练包含相同的强制站立段。

指定 `--vx` 切换为恒定速度，`--steps` 可限制回放步数：

```bash
python humanoid/scripts/play_amp.py --task=x1_amp --load_run='<run>' --checkpoint=20 --num_envs=1 --steps=1000 --vx=0.3 --vy=0 --yaw_rate=0
```

服务器无窗口回放时添加 `--headless`。回放关闭观测噪声、推力和参考状态初始化，从配置的默认姿态开始；每隔 100 步输出一次状态。新策略应使用此脚本；旧的 DH 导出与 12 关节 sim2sim 脚本不适用于本任务。

导出普通 MLP TorchScript 策略及对应控制/观测说明：

```bash
python humanoid/scripts/play_amp.py --task=x1_amp --load_run='<run>' --checkpoint=20 --steps=100 --headless --no_video --export_policy=logs/x1_amp/policy.pt
```

同时生成 `policy.pt.json`，保存仿真器实际关节顺序、默认关节角、PD 参数、动作比例及观测缩放。导出策略输入为 480 维，输出为 29 维。

本地模型可通过 `--checkpoint_path=/path/to/model_6000.pt` 直接加载，不恢复优化器；现有 `--load_run/--checkpoint` 方式继续可用。

## 算法和尺寸

| 项目 | 默认设置 |
| --- | --- |
| 物理步长 / 策略周期 | 0.002 s / 0.02 s（50 Hz） |
| 动作 | 29 维，目标角 = 默认角 + 0.25 × 裁剪后的动作 |
| 策略单帧 | 指令 3 + 基座角速度 3 + 重力投影 3 + 关节角偏差 29 + 关节速度 29 + 上次动作 29 = 96 |
| 策略输入 | 按从旧到新的顺序拼接 5 帧，480 维 |
| 价值输入 | 无噪声的策略单帧 + 基座线速度 3，99 维 |
| AMP 单帧 | 基座高度 1 + 重力 3 + 局部线/角速度 6 + 关节角/速度 58 + 局部关键点 18 = 86 |
| 判别器输入 | 连续 3 帧，258 维 |
| 专家/策略判别目标 | +1 / -1（LSGAN） |
| 风格奖励 | `dt * 2 * max(0, 1 - (D-1)^2/4)` |
| 总奖励 | `0.5 * task_reward + 0.5 * style_reward` |

判别器使用在线统计归一化、策略回放池和专家样本梯度惩罚。任务奖励包括速度跟踪、存活、姿态、动作变化、能耗和限位等项，不再使用原 12 关节的手工摆腿参考奖励。参考状态初始化同时设置根姿态、根速度、29 个关节位置/速度以及历史 AMP 特征。

环境在自动重置前保留终止 AMP 状态和价值网络输入。超时通过终止状态价值进行 bootstrap；判别器不会把重置后的姿态当作上个 episode 的末帧。

主要参数位于 `humanoid/envs/x1/x1_amp_config.py`。初始默认姿态使用四段规范化轨迹的平均关节角；PD 增益参考 AMP 工程，扭矩/速度限位使用当前 URDF。先在平地、无推力随机化的基线上训练，再逐步加入扰动。当前参考动作集中于慢走与站走转换，默认指令范围也相应收窄。

## exp_1.1 独立修正任务

任务名为 `x1_amp_exp11`，实现位于 `x1_amp_exp11_config.py` 和 `x1_amp_exp11_env.py`。新增逐关节硬限位内的PD目标、原始目标越界代价、腰/上肢/下肢侧向姿态软约束，以及50%站姿/50%RSI初始化。站姿分支的1.5秒零指令缓冲由训练和play共用，初始化不再隐含执行一次指向平均步行姿态的零动作。旧的 `x1_amp` 保留原控制与schema=1；新任务使用schema=2，不能直接恢复旧控制方案的模型。

验证发现原参考站立帧的碰撞脚底穿地约3cm。仅新任务启用碰撞网格脚底高度对齐，将最低脚底置于地面上方2mm并重算根线速度；原始轨迹文件、关节角及旧任务不变。

**当前状态：功能链路通过，固定PD的5秒站立门禁未通过；2026-10-26用户明确决定暂不处理此问题，按现有配置启动6000轮训练。** 参考静止帧及几何校准候选均有失稳，静态验收失败记录保留；正式训练结果另行评估。详细结果与本次启动决定见 `czy/exp1/exp1.md` 的exp_1.1记录。

可复现的诊断入口（需Isaac Gym环境）：

```bash
python humanoid/scripts/validate_exp11.py --task=x1_amp_exp11 --headless --num_envs=3 --phase=static
python humanoid/scripts/validate_exp11.py --task=x1_amp_exp11 --headless --num_envs=64 --phase=smoke
python humanoid/scripts/validate_exp11.py --task=x1_amp_exp11 --headless --num_envs=8 --phase=timing
```

固定PD站姿验收录像（无需模型）：

```bash
xvfb-run -a python humanoid/scripts/record_standing.py --task=x1_amp_exp11 --headless --seed=5 --seeds 5 7 11 --output_dir=logs/standing_acceptance
```

每个种子同屏录制3个关节扰动样本，各仿真5秒；1920×1080、25fps，片头和片尾各停留1秒。固定目标为B1第11帧站姿，沿用原PD增益与终止条件。录像专用入口暂停自动重置，并锁定首次失败时间；失败后的画面只展示倒下过程。输出MP4、诊断CSV和元数据，不能把它当作策略回放或正式训练三件套。

新任务回放使用 `play_amp.py --task=x1_amp_exp11 --checkpoint_path=<该任务模型>`；`--reset_mode=standing` 与 `--reset_mode=rsi` 分别检查两类初始化。CSV增加raw/effective目标、逐关节限幅次数、初始化类型和缓冲剩余步数。

## 训练日志口径

沿用原 runner 的多行展示与 TensorBoard 标签：`Loss/`、`Policy/`、`Perf/`、`Train/`、`Episode/`、`Observation/`。`Train/mean_reward` 是最近100个已结束回合的累计混合奖励均值（0.5任务+0.5风格），不包含超时bootstrap的价值补偿；`Train/mean_episode_length` 是同一批回合的实际长度均值。尚无回合结束时不写这两个指标。逐项任务奖励沿用环境 `Episode/rew_*` 的统计口径，除以配置的最大回合时长，并非除以实际存活时间。

新增AMP每步任务/风格奖励、回合任务/风格累计奖励、判别器损失/梯度惩罚、专家与策略分数、熵、KL和终止比例。原有 `AMP/*` 曲线继续保留；其中历史 `AMP/episode_length` 仍指当前回合年龄，新图使用明确的 `AMP/current_episode_age`，完整回合长度使用 `Train/mean_episode_length`。采样耗时、学习耗时、FPS、总步数、总耗时和ETA恢复显示；总步数与耗时从本次runner启动计，旧checkpoint不包含此前的日志累计器。观测统计按当前全部输入维度记录，AMP不使用原DH状态估计器，因此不输出该模块的损失。

## 验证

云端任务查询、训练曲线获取与服务器回放归档工具见 [scripts/amp_tools](scripts/amp_tools/README.md)。历史验证及正式训练记录位于 `czy/exp1/artifacts/exp_2/initial_validation/` 和 `czy/exp1/artifacts/exp_2/training/`；原 `czy/validation/` 已清空。

不依赖 Isaac Gym 的回归测试：

```bash
python -m unittest discover -s tests -v
```

覆盖关节重排、四元数、历史边界、归一化、回放环形写入、PPO/判别器反向传播、检查点恢复与超时价值估计。`validate_amp.py` 额外检查真正的 GPU 仿真关节/关键点坐标、观测尺寸和批量自动重置。

短程训练通过只说明数据、仿真和算法更新链路可运行，不代表策略已经收敛或具备稳定行走能力。正式训练需要继续观察 episode 长度、速度跟踪和回放步态。
