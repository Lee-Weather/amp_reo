# AMP 实验工具

这些本机工具从原 `czy/validation/` 迁入，不参与训练算法。以下命令均从项目根目录执行。

| 脚本 | 用途 |
| --- | --- |
| `flux_ops.py` | 调用已安装的Flux CLI，对返回的凭据与签名URL脱敏 |
| `flux_status.py` | 按指定任务状态文件查询训练状态，更新旁边的脱敏日志 |
| `server_status.py` | 查询记录的201.5训练进程、实际轮数、恢复优化器凭据和保存记录 |
| `replay_dependency.py` | 在服务器等待指定训练进程结束，核对最终保存凭据后准备回放副本 |
| `launch_fixed_amp.py` | 用输入输出同目录规则提交训练，核对账号、源模型和远端提交 |
| `fetch_amp_checkpoint.py` | 下载实际训练模型，对照保存凭据校验大小、哈希、轮数和模型约定 |
| `fetch_server_checkpoint.py` | 按服务器保存凭据取回恢复模型，核对双方优化器步数及归一化器延续 |
| `fetch_replay_curves.py` | 获取指定训练任务的AMP曲线，保存到明确指定的附件路径 |
| `replay_archive.py` | 在201.5服务器新目录部署当前代码并启动完整回放 |
| `collect_replay_archive.py` | 取回完成的回放，校验视频/CSV/模型并归档、生成诊断汇总 |
| `analyze_replay.py` | 共用CSV评估：过渡段剔除、速度、姿态、终止及原始/有效目标限位 |
| `analyze_gait.py` | 离线对照参考与策略的同腿周期、关节摆幅、碰撞脚底高度、PD目标与实际运动 |
| `analyze_standing.py` | 验证后段零指令，量化姿态/膝踝/PD目标晃动并出图 |
| `analyze_amp_features.py` | 重建并验证CSV的AMP历史，检查冻结判别器对各类特征的敏感度 |

Flux工具使用本机已安装CLI，继续从已忽略的 `czy/skills/flux-cli/api_key.json` 读取凭据，不将凭据复制到本目录。SSH工具使用Windows自带OpenSSH和密钥登录。

查询已完成训练，状态和日志保存在实验附件目录：

```powershell
python scripts/amp_tools/flux_status.py --state czy/exp1/exp1.3_training.json
python scripts/amp_tools/fetch_replay_curves.py --task-id TASK_20261026_229 --account limxmts0d1flmto8rp@uberip.com --output czy/exp1/exp1.3_evaluation/training_curves.json
```

`fetch_replay_curves.py` 支持 `--account`、`--end-time`；`flux_status.py` 根据状态文件中的账号选择凭据。工具不会默认往 `analysis` 写文件。

回放工具读取模型内部iter，按`model_{iter}.pt`打包、校验和归档，支持中期检查点与最终模型；视频仍为最新play的六段60秒、1080p/25fps配置。修改回放配置时需同步其校验条件。以下为后续实验调用示例，执行前须已有用户授权的服务器回放任务：

```powershell
python scripts/amp_tools/replay_archive.py --experiment <实验编号> --model <模型绝对路径> --resource_source <服务器资源目录> --task <技术task> --training-task-id <实际Task-ID> --record_dir czy/exp1/<实验编号>_evaluation --check_rsi
python scripts/amp_tools/collect_replay_archive.py --experiment <实验编号> --model <同一模型绝对路径> --record_dir czy/exp1/<实验编号>_evaluation
```

exp_1.1等独立AMP任务必须显式指定任务与训练来源，可用专用记录目录避免恢复已清理的artifacts目录：

```powershell
python scripts/amp_tools/replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --resource_source /home/robot/czy/exp1/exp_20261026_105426 --task x1_amp_exp11 --training-task-id TASK_20261026_051 --record_dir czy/exp1/exp_1.1_evaluation --check_rsi
python scripts/amp_tools/collect_replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --record_dir czy/exp1/exp_1.1_evaluation
```

上述部署命令不可对已有任务重复执行。`--primary_only`可在辅助验证期间先取回主三件套，完成后用`--checks_only`补齐同一报告；辅助CSV在临时目录处理后删除，服务器保留原始证据。允许交付目录预先存在同一模型，拒绝覆盖已有回放。笔记依据实际诊断结果更新，不预设“部分达标”。速度稳态窗口剔除指令切换、重置、缓冲结束后2秒；姿态统计剔除回合前100步、终止步和根高度≤0.45m样本。

可追加`--check_speeds -0.2 0.1 0.65`检查后退、慢走和速度上限，各用seed5站姿初始化、恒速1000步（20秒）、不录视频。`--speed_check_steps`可增加时长，不能少于1000步。收集工具按记录的步数验证元数据和CSV；主视频及六段回放保持不变。这些检查的速度指标见各自`evaluation.segments`与`walking`，不能套用0.4/0.6专用指标；摔倒次数仍按完整记录计算。

部署脚本使用系统临时目录打包，取回工具也使用系统临时目录校验，结束后自动清理。已有任务状态或归档时会拒绝覆盖，不要重复启动。

结果位置：

- `czy/data/{实验编号}/`：只存模型、视频、CSV三件套。
- `czy/exp1/{实验编号}_evaluation/`：诊断图、训练曲线、元数据、日志和摘要。
- `czy/exp1/{实验编号}_training.json/log`：训练状态和脱敏日志。

已清理的artifacts目录不再恢复。当前实验exp1.5，下一轮exp1.6；当前归档还包括exp1.2（原exp_2）与exp1.3（原exp_3），未经用户明确同意不得进入exp2。后续只用seed=5；不传`--extra_seeds`，`--check_rsi`也只检查seed=5。技术task名x1_amp_exp3保留，历史启动命令/分支不因标签变化而改写。

原 `czy/validation/` 已清空；部署压缩包、短测导出模型、参考工程快照和一次性脚本不保留。

exp_1.1小碎步专项分析（CPU，不启动仿真或修改模型）：

```powershell
python scripts/amp_tools/analyze_gait.py --model czy/data/exp_1.1/model_6000.pt --csv czy/data/exp_1.1/isaac_diag.csv --output_dir czy/exp1/exp_1.1_evaluation
```

该工具目前明确使用exp_1.1六段回放的稳态时间窗与B3连续前进参考；换回放方案时必须调整窗口。输出gait_comparison.json/png；“同腿周期前进距离”不是落脚点步长，最低脚底几何高度不是接触力。

exp1.3训练任务为`x1_amp_exp3`。历史上用户指定从零训练6000轮，原始启动命令如下（已完成，不重复启动）：

```powershell
python humanoid/scripts/train.py --task=x1_amp_exp3 --headless --num_envs=4096 --seed=5 --max_iterations=6000 --run_name=exp_3
```

仅训练vx=[-0.2,0.65]，vy/wz固定0；线速度sigma=0.09，角速度sigma=0.25。
PPO学习率初值/最大值1e-4、最小值1e-5，desired_kl=0.02；每小批次更新前计算解析KL(old||current)，超过0.03停止本轮后续更新，末次更新后另计算全rollout KL。学习率每轮调整一次（最大观测KL>0.03则除1.5，<0.01则乘1.5并限制上下界）；一次更新仍可能超限，不是硬KL约束。
AMP风格尺度10，任务/风格混合0.5/0.5；`AMP/style_score`为不乘尺度和dt的分数，回放CSV同名列。原runner完整回合统计及日志保留，追加KL最大值/提前停止/实际更新次数/学习率。
正式任务一次训练6000轮，每100轮保存检查点；原方案的3000+3000已按用户指令改为连续6000轮。runner的max_iterations是本次调用新增轮数，后续若恢复需按实际剩余轮数设置。判别器强度、历史帧数、PD增益、action_scale、LPF及抬脚奖励保持原设置。
回放请显式传`--task x1_amp_exp3`，并为各检查点指定不同experiment和record_dir；不可覆盖正式历史三件套。

零速晃动专项分析（固定六段回放的后段零指令窗口）：

```powershell
python scripts/amp_tools/analyze_standing.py --csv czy/data/exp1.3/isaac_diag.csv --output_dir czy/exp1/exp1.3_evaluation
```

输出standing_analysis.json/png，P95-P05为摆幅，不是姿态均值偏移；工具拒绝非零指令或包含重置的时间窗。换回放时间表时需调整窗口。

判别器诊断可用`python scripts/amp_tools/analyze_amp_features.py --model <模型> --csv <CSV> --output_dir <实验附件目录>`，仅适用于当前29DOF、2mm脚底对齐方案。它先核对重建风格分数与CSV记录，再将静止/行走样本分别与匹配指令的真实参考比较。输出`amp_feature_analysis.json`；特征替换可能破坏物理一致性，局部梯度或分数变化只说明判别器敏感度，不能证明某个训练改动有效。


exp1.4独立任务`x1_amp_exp14`：每次重采样20%精确零指令，0.5秒渐增静止代价，真实低运动专家混合与指令匹配RSI；继承29DOF/50Hz和原PPO KL保护。正式热启动以exp1.3/model_6000为源，初始学习率经GPU验证修正为1e-5，双方优化器重建；新增1000轮到model_7000。原任务120完成但模型未取回，保留czy/exp1/exp1.4_training.json；修复交付路径后的正式重试为TASK_20261027_166，状态文件czy/exp1/exp1.4_training_retry1.json，不能混用两次任务的模型和结果。

`--warm_start`与`--resume`互斥；前者继承权重和iter、新建优化器，后者恢复完整优化器状态。指定`--warm_start_sha256`核对源模型；云端临时URL回退只保存在运行参数内，日志与状态文件必须脱敏。

云端固定checkpoint目录：训练可显式传`--checkpoint_dir=/workspace/amp_reo/logs/x1_amp_exp14/exported_data/upload_check`，并将`--warm_start`指向同目录的model_6000.pt。平台`checkPointMountPath`也设为对应的仓库相对目录。该目录只允许已有输入模型，拒绝覆盖既有训练结果；不要把不同任务都输出到同一个旧目录。CHECKPOINT_INPUT_READY为输入副本事件，CHECKPOINT_SAVED为本地保存，只有实际模型列表和下载验证才证明上传成功。

exp1.5候选任务`x1_amp_exp15`保持相同控制与观测约定，增加速度相关跟踪容差、零速腿部约束和移动脚底拖地代价；零指令的AMP乘数在0.5秒内降到0.2。abs(vx指令)>0.05时另乘指令方向的实际速度/指令速度并限制为[0,1]，避免原地获取完整运动AMP奖励，不为超速加分。`style_score`仍是原始判别器分数，`style_reward`为实际门控后的风格贡献，CSV另有`style_reward_raw`和`amp_style_multiplier`；主runner原有回合日志不变。当前完整代码36项CPU检查及7000→7005的4096环境GPU预检通过，另核对了零速/0.1回放的风格和混合奖励公式。正式TASK_20261027_186从实际7000新增1000轮到8000，首个7100已下载校验；预检7005不作为正式训练源或最终成果。

任务186于7263附近被平台终止，原因未确认；已取回实际7200并在201.5从7200完整恢复剩余800轮到8000。当前查询命令是`python scripts/amp_tools/server_status.py --state czy/exp1/exp1.5_training_recovery.json`；旧exp1.5_training.json记录云端状态6，不能代替服务器进程状态。恢复使用`--resume`保留双方Adam状态，CHECKPOINT_RESTORED确认原step4000/800和学习率；后续服务器模型按实际保存记录经SCP取回并核对哈希，不期待旧云端模型列表出现8000。

先刷新服务器状态，再用`python scripts/amp_tools/fetch_server_checkpoint.py --state czy/exp1/exp1.5_training_recovery.json --reference-model czy/exp1/exp1.5_evaluation/recovery/model_7200.pt --output-dir czy/data/exp1.5 --iteration 8000`取回最终模型。工具要求实际训练进程结束、存在最终保存凭据，验证文件大小/hash、iter、控制元数据、完整恢复来源和权重有限性，并要求双方优化器计数与归一化器count相对7200均增加；不会用重新清空优化器的模型冒充完整恢复。中途7300已经按此规则验证通过。最后照原流程回放并补齐严格三件套。

exp1.5最终回放已排队：状态为`czy/exp1/exp1.5_evaluation/replay_job.json`，服务器`/home/robot/czy/exp1/exp_20261027_131606`，等待恢复训练PID3078307。训练仍存活时不启动GPU回放；进程结束还必须有SERVER_RECOVERY_OK和8000保存凭据，且源文件大小/hash一致才复制模型并录视频。训练异常退出则中止回放，不把中期模型当最终模型。不要再调用replay_archive.py重复部署；完成后取回实际8000，再用collect_replay_archive.py的primary_only/checks_only分阶段收集。排队时checkpoint_sha256为空是因为最终文件尚不存在，收集器会在主视频完成后读取prepared_checkpoint.json，并与下载模型及训练来源逐项核对后填入，不能手填猜测哈希。


exp1.5的8000模型和最终三件套已取回，主/RSI均无跌倒，但后退、低速与右脚离地未达标。完整验收见czy/exp1/exp1.5_evaluation/acceptance_summary.json。exp1.6从真实8000热启动，增加运动AMP速度误差质量因子与四髋膝pitch运动条件std下限0.25，保持零速原方差、原控制与日志展示。45项CPU检查、8000→8005的4096/seed5 GPU短训及三项短回放公式核对均通过；这些不替代正式步态验收。

201.5新任务用launch_server_amp.py：显式指定experiment/task/model/sha256/iterations/state，固定4096/seed5。工具仅传humanoid Python、setup.py及输入模型，复用已验证resources；拒绝GPU已有计算进程、已有state及已有远程目录。预检额外加--preflight。状态在启动前持久化，启动观察超时必须检查同目录和进程，禁止自动重启。server_status.py兼容旧恢复日志和新任务记录的log_file/success_marker；fetch_server_checkpoint.py按resume_mode分别验证完整恢复动量延续或热启动新Adam计数范围、真实输入SHA与控制元数据。


exp1.6正式状态：czy/exp1/exp1.6_training.json，源8000、目标9000，代码dea3d4f；server_status.py按该文件轮询同一进程。最终回放已通过queue_server_replay.py排队，状态czy/exp1/exp1.6_evaluation/replay_job.json；禁止重复部署。工具复制正式训练的代码，等待记录的launcher退出、成功标记、保存凭据和GPU空闲后执行主六段视频/RSI及五项恒速检查。训练完成后先按真实8000参考取回9000，再用collect_replay_archive.py分阶段收集，最终czy/data/exp1.6只保留模型、视频、CSV三项。


完整验收用summarize_amp_acceptance.py，指定model/csv/record-dir及criteria-version=exp1.6；输入应是完成收集的严格三件套及replay_summary/replay_metadata。脚本重新计算真实脚底、完整周期偏航/侧向速度、静止摆幅和持续0.2秒刹停确认，并检查全部站姿/RSI/速度条件、收集哈希及文件数量。生成acceptance_summary.json；未完成视频自然度复核时accepted始终false。复核JSON通过--visual-review指定，须含status=passed、checkpoint_sha256、video_sha256及具体observations，且哈希匹配当前视频与模型。旧报告缺少新收集哈希时只标记证据缺失，不把旧文件判为损坏。


exp1.6已正常完成9000并收集完整三件套及六项辅助检查，见czy/exp1/exp1.6_evaluation/acceptance_summary.json；accepted=false，后退仍停住且正向有持续偏航。原训练/回放句柄均已结束，不要重启已完成的任务。下一轮编号exp1.7，待实现方案和预检；保留9000为候选续训源、8000为安静站姿基准。

fetch_server_curves.py现通过远程临时JSON文件及校验SCP快照取数，避免大体积SSH stdout超时。默认不导出960个Observation逐维tag，可用--include-observations包含；原服务器TensorBoard日志完整保留，原runner展示不变。临时快照校验取回后按确切路径清理，不增加最终三件套文件。
