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
| `replay_archive.py` | 在201.5服务器新目录部署指定训练提交并启动完整回放 |
| `fetch_final_amp.py` | 对照最终保存记录取回模型，验证来源、完整DH推理及双方Adam状态 |
| `watch_amp_finals.py` | 等待明确指定的云任务，完成且最终模型上传后才下载；不会重启训练 |
| `collect_training_curves.py` | 采集精确训练曲线，按相同更新窗口汇总完成回合和分组指标 |
| `collect_replay_archive.py` | 取回完成的回放，校验视频/CSV/模型并归档、生成诊断汇总 |
| `analyze_replay.py` | 共用CSV评估：过渡段剔除、速度、姿态、终止及原始/有效目标限位 |
| `analyze_gait.py` | 离线对照参考与策略的同腿周期、关节摆幅、碰撞脚底高度、PD目标与实际运动 |
| `analyze_standing.py` | 验证后段零指令，量化姿态/膝踝/PD目标晃动并出图 |
| `analyze_amp_features.py` | 重建并验证CSV的AMP历史，检查冻结判别器对各类特征的敏感度 |

Flux工具使用本机已安装CLI，继续从已忽略的 `czy/skills/flux-cli/api_key.json` 读取凭据，不将凭据复制到本目录。SSH工具使用Windows自带OpenSSH和密钥登录。

查询已完成训练，状态和日志保存在实验附件目录：

```powershell
python scripts/amp_tools/flux_status.py --state czy/exp1/records/exp1.3_training.json
python scripts/amp_tools/fetch_replay_curves.py --task-id TASK_20261026_229 --account limxmts0d1flmto8rp@uberip.com --output czy/exp1/records/exp1.3_evaluation/training_curves.json
```

`fetch_replay_curves.py` 支持 `--account`、`--end-time`；`flux_status.py` 根据状态文件中的账号选择凭据。工具不会默认往 `analysis` 写文件。

回放工具读取模型内部iter，按`model_{iter}.pt`打包、校验和归档，支持中期检查点与最终模型；视频仍为最新play的六段60秒、1080p/25fps配置。修改回放配置时需同步其校验条件。以下为后续实验调用示例，执行前须已有用户授权的服务器回放任务：

```powershell
python scripts/amp_tools/replay_archive.py --experiment <实验编号> --model <模型绝对路径> --resource_source <服务器资源目录> --task <技术task> --training-task-id <实际Task-ID> --record_dir czy/exp1/records/<实验编号>_evaluation --check_rsi
python scripts/amp_tools/collect_replay_archive.py --experiment <实验编号> --model <同一模型绝对路径> --record_dir czy/exp1/records/<实验编号>_evaluation
```

exp_1.1等独立AMP任务必须显式指定任务与训练来源，可用专用记录目录避免恢复已清理的artifacts目录：

```powershell
python scripts/amp_tools/replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --resource_source /home/robot/czy/exp1/exp_20261026_105426 --task x1_amp_exp11 --training-task-id TASK_20261026_051 --record_dir czy/exp1/records/exp_1.1_evaluation --check_rsi
python scripts/amp_tools/collect_replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --record_dir czy/exp1/records/exp_1.1_evaluation
```

上述部署命令不可对已有任务重复执行。`--primary_only`可在辅助验证期间先取回主三件套，完成后用`--checks_only`补齐同一报告；辅助CSV、元数据和日志校验后压缩为evaluation目录中的`auxiliary_replays.tar.gz`，不加入三件套目录。允许交付目录预先存在同一模型，拒绝覆盖已有回放。笔记依据实际诊断结果更新，不预设“部分达标”。速度稳态窗口剔除指令切换、重置、缓冲结束后2秒；姿态统计剔除回合前100步、终止步和根高度≤0.45m样本。

完成云端实验的回放应显式传`--code-commit <训练的完整40位SHA>`；工具从Git对象读取该提交，忽略之后的工作区修改，只打包humanoid Python、setup.py和模型。服务器启动仿真前逐SHA核对上述文件及本地67项X1模型/轨迹资源，证据写入`deployment_verification.json`并由收集器复核。任务支持`exp1.13t`等用户指定的t后缀，回放仍仅seed=5。GPU检查允许已观察到的低于512MiB的NoMachine桌面进程，其余计算进程均阻止启动；不会结束其他进程。

正向课程实验可传`--primary-vx 0.3`，主视频改为站姿起步、单一0.3m/s请求、3000步约60秒；省略该参数时仍为历史六段时间表。辅助检查使用`--check_speeds 0 --check_rsi_speeds 0.3 --speed_check_steps 3000`可分别取得60秒零速与RSI起步CSV。收集器按记录的主时间表校验分段，不套用历史0.4/0.6指标判定新课程。

单一前进回放的CPU碎步分析使用`analyze_forward_gait.py --model <模型> --csv <主CSV> --output-dir <evaluation>`，不启动仿真。默认分析4–18、24–38、44–58秒，拒绝跨重置/缓冲/指令切换的窗口；从checkpoint重建实际+0.3m/s缩放参考，校验参考文件及代码哈希、脚端FK与CSV一致。输出`forward_gait_analysis.json`和`forward_gait_comparison.png`。频率是关节主频，脚前后摆幅是相对骨盆P95-P05，均不能直接称作落脚步频或步长；参考脚底高度是接地后的运动学目标。

最终模型收集使用`python scripts/amp_tools/fetch_final_amp.py --state <训练状态JSON> --via-server`，要求平台状态5、实际训练commit匹配、精确文件名`model_<最终轮数>.pt`已上传，且下载字节数/SHA与CHECKPOINT_SAVED相同；输入副本不能冒充最终模型。`--via-server`只经已授权201.5主机临时下载后SCP取回，不输出签名URL。训练曲线可用`python scripts/amp_tools/collect_training_curves.py --states <状态JSON> [<另一状态JSON>]`收集；运行中始终使用precise，同一last100窗口结束轮次一致。

可追加`--check_speeds -0.2 0.1 0.65`检查后退、慢走和速度上限，各用seed5站姿初始化、恒速1000步（20秒）、不录视频。`--speed_check_steps`可增加时长，不能少于1000步。收集工具按记录的步数验证元数据和CSV；主视频及六段回放保持不变。这些检查的速度指标见各自`evaluation.segments`与`walking`，不能套用0.4/0.6专用指标；摔倒次数仍按完整记录计算。

部署脚本使用系统临时目录打包，取回工具也使用系统临时目录校验，结束后自动清理。已有任务状态或归档时会拒绝覆盖，不要重复启动。

结果位置：

- `czy/data/{实验编号}/`：只存模型、视频、CSV三件套。
- `czy/exp1/records/{实验编号}_evaluation/`：诊断图、训练曲线、元数据、日志和摘要。
- `czy/exp1/records/{实验编号}_training.json/log`：训练状态和脱敏日志。

已清理的artifacts目录不再恢复。当前实验exp1.5，下一轮exp1.6；当前归档还包括exp1.2（原exp_2）与exp1.3（原exp_3），未经用户明确同意不得进入exp2。后续只用seed=5；不传`--extra_seeds`，`--check_rsi`也只检查seed=5。技术task名x1_amp_exp3保留，历史启动命令/分支不因标签变化而改写。

原 `czy/validation/` 已清空；部署压缩包、短测导出模型、参考工程快照和一次性脚本不保留。

exp_1.1小碎步专项分析（CPU，不启动仿真或修改模型）：

```powershell
python scripts/amp_tools/analyze_gait.py --model czy/data/exp_1.1/model_6000.pt --csv czy/data/exp_1.1/isaac_diag.csv --output_dir czy/exp1/records/exp_1.1_evaluation
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
python scripts/amp_tools/analyze_standing.py --csv czy/data/exp1.3/isaac_diag.csv --output_dir czy/exp1/records/exp1.3_evaluation
```

输出standing_analysis.json/png，P95-P05为摆幅，不是姿态均值偏移；工具拒绝非零指令或包含重置的时间窗。换回放时间表时需调整窗口。

判别器诊断可用`python scripts/amp_tools/analyze_amp_features.py --model <模型> --csv <CSV> --output_dir <实验附件目录>`，仅适用于当前29DOF、2mm脚底对齐方案。它先核对重建风格分数与CSV记录，再将静止/行走样本分别与匹配指令的真实参考比较。输出`amp_feature_analysis.json`；特征替换可能破坏物理一致性，局部梯度或分数变化只说明判别器敏感度，不能证明某个训练改动有效。


exp1.4独立任务`x1_amp_exp14`：每次重采样20%精确零指令，0.5秒渐增静止代价，真实低运动专家混合与指令匹配RSI；继承29DOF/50Hz和原PPO KL保护。正式热启动以exp1.3/model_6000为源，初始学习率经GPU验证修正为1e-5，双方优化器重建；新增1000轮到model_7000。原任务120完成但模型未取回，保留czy/exp1/records/exp1.4_training.json；修复交付路径后的正式重试为TASK_20261027_166，状态文件czy/exp1/records/exp1.4_training_retry1.json，不能混用两次任务的模型和结果。

`--warm_start`与`--resume`互斥；前者继承权重和iter、新建优化器，后者恢复完整优化器状态。指定`--warm_start_sha256`核对源模型；云端临时URL回退只保存在运行参数内，日志与状态文件必须脱敏。

云端固定checkpoint目录：训练可显式传`--checkpoint_dir=/workspace/amp_reo/logs/x1_amp_exp14/exported_data/upload_check`，并将`--warm_start`指向同目录的model_6000.pt。平台`checkPointMountPath`也设为对应的仓库相对目录。该目录只允许已有输入模型，拒绝覆盖既有训练结果；不要把不同任务都输出到同一个旧目录。CHECKPOINT_INPUT_READY为输入副本事件，CHECKPOINT_SAVED为本地保存，只有实际模型列表和下载验证才证明上传成功。

exp1.5候选任务`x1_amp_exp15`保持相同控制与观测约定，增加速度相关跟踪容差、零速腿部约束和移动脚底拖地代价；零指令的AMP乘数在0.5秒内降到0.2。abs(vx指令)>0.05时另乘指令方向的实际速度/指令速度并限制为[0,1]，避免原地获取完整运动AMP奖励，不为超速加分。`style_score`仍是原始判别器分数，`style_reward`为实际门控后的风格贡献，CSV另有`style_reward_raw`和`amp_style_multiplier`；主runner原有回合日志不变。当前完整代码36项CPU检查及7000→7005的4096环境GPU预检通过，另核对了零速/0.1回放的风格和混合奖励公式。正式TASK_20261027_186从实际7000新增1000轮到8000，首个7100已下载校验；预检7005不作为正式训练源或最终成果。

任务186于7263附近被平台终止，原因未确认；已取回实际7200并在201.5从7200完整恢复剩余800轮到8000。当前查询命令是`python scripts/amp_tools/server_status.py --state czy/exp1/records/exp1.5_training_recovery.json`；旧exp1.5_training.json记录云端状态6，不能代替服务器进程状态。恢复使用`--resume`保留双方Adam状态，CHECKPOINT_RESTORED确认原step4000/800和学习率；后续服务器模型按实际保存记录经SCP取回并核对哈希，不期待旧云端模型列表出现8000。

先刷新服务器状态，再用`python scripts/amp_tools/fetch_server_checkpoint.py --state czy/exp1/records/exp1.5_training_recovery.json --reference-model czy/exp1/records/exp1.5_evaluation/recovery/model_7200.pt --output-dir czy/data/exp1.5 --iteration 8000`取回最终模型。工具要求实际训练进程结束、存在最终保存凭据，验证文件大小/hash、iter、控制元数据、完整恢复来源和权重有限性，并要求双方优化器计数与归一化器count相对7200均增加；不会用重新清空优化器的模型冒充完整恢复。中途7300已经按此规则验证通过。最后照原流程回放并补齐严格三件套。

exp1.5最终回放已排队：状态为`czy/exp1/records/exp1.5_evaluation/replay_job.json`，服务器`/home/robot/czy/exp1/exp_20261027_131606`，等待恢复训练PID3078307。训练仍存活时不启动GPU回放；进程结束还必须有SERVER_RECOVERY_OK和8000保存凭据，且源文件大小/hash一致才复制模型并录视频。训练异常退出则中止回放，不把中期模型当最终模型。不要再调用replay_archive.py重复部署；完成后取回实际8000，再用collect_replay_archive.py的primary_only/checks_only分阶段收集。排队时checkpoint_sha256为空是因为最终文件尚不存在，收集器会在主视频完成后读取prepared_checkpoint.json，并与下载模型及训练来源逐项核对后填入，不能手填猜测哈希。


exp1.5的8000模型和最终三件套已取回，主/RSI均无跌倒，但后退、低速与右脚离地未达标。完整验收见czy/exp1/records/exp1.5_evaluation/acceptance_summary.json。exp1.6从真实8000热启动，增加运动AMP速度误差质量因子与四髋膝pitch运动条件std下限0.25，保持零速原方差、原控制与日志展示。45项CPU检查、8000→8005的4096/seed5 GPU短训及三项短回放公式核对均通过；这些不替代正式步态验收。

201.5新任务用launch_server_amp.py：显式指定experiment/task/model/sha256/iterations/state，固定4096/seed5。工具仅传humanoid Python、setup.py及输入模型，复用已验证resources；拒绝GPU已有计算进程、已有state及已有远程目录。预检额外加--preflight。状态在启动前持久化，启动观察超时必须检查同目录和进程，禁止自动重启。server_status.py兼容旧恢复日志和新任务记录的log_file/success_marker；fetch_server_checkpoint.py按resume_mode分别验证完整恢复动量延续或热启动新Adam计数范围、真实输入SHA与控制元数据。


exp1.6正式状态：czy/exp1/records/exp1.6_training.json，源8000、目标9000，代码dea3d4f；server_status.py按该文件轮询同一进程。最终回放已通过queue_server_replay.py排队，状态czy/exp1/records/exp1.6_evaluation/replay_job.json；禁止重复部署。工具复制正式训练的代码，等待记录的launcher退出、成功标记、保存凭据和GPU空闲后执行主六段视频/RSI及五项恒速检查。训练完成后先按真实8000参考取回9000，再用collect_replay_archive.py分阶段收集，最终czy/data/exp1.6只保留模型、视频、CSV三项。


完整验收用summarize_amp_acceptance.py，指定model/csv/record-dir及criteria-version=exp1.6；输入应是完成收集的严格三件套及replay_summary/replay_metadata。脚本重新计算真实脚底、完整周期偏航/侧向速度、静止摆幅和持续0.2秒刹停确认，并检查全部站姿/RSI/速度条件、收集哈希及文件数量。生成acceptance_summary.json；未完成视频自然度复核时accepted始终false。复核JSON通过--visual-review指定，须含status=passed、checkpoint_sha256、video_sha256及具体observations，且哈希匹配当前视频与模型。旧报告缺少新收集哈希时只标记证据缺失，不把旧文件判为损坏。


exp1.6已正常完成9000并收集完整三件套及六项辅助检查，见czy/exp1/records/exp1.6_evaluation/acceptance_summary.json；accepted=false，后退仍停住且正向有持续偏航。原训练/回放句柄均已结束，不要重启已完成的任务。下一轮编号exp1.7，待实现方案和预检；保留9000为候选续训源、8000为安静站姿基准。

fetch_server_curves.py现通过远程临时JSON文件及校验SCP快照取数，避免大体积SSH stdout超时。默认不导出960个Observation逐维tag，可用--include-observations包含；原服务器TensorBoard日志完整保留，原runner展示不变。临时快照校验取回后按确切路径清理，不增加最终三件套文件。


冻结策略的时间相关探索诊断（只读策略，4096环境、seed=5、GPU）：

```powershell
python scripts/amp_tools/run_temporal_probe.py launch --training-state czy/exp1/records/exp1.8_training.json --job czy/exp1/records/exp1.8_evaluation/temporal_probe_job.json
python scripts/amp_tools/run_temporal_probe.py collect --job czy/exp1/records/exp1.8_evaluation/temporal_probe_job.json
```

只在GPU空闲时启动，已有job拒绝重启；收集前核对原PID终止和TEMPORAL_PROBE_OK。该诊断复制实际exp1.8生产代码和11000模型，核对SHA，不修改生产采样或更新权重。四组各1024环境，空间交错分配：原IID、六髋膝踝pitch额外2度白噪声、同幅值保持10/20步；后三组名义边际方差相同，实际扰动RMS另报。无传感噪声/推力，保留真实站姿扰动和1.5秒缓冲；任何重置均从首回合成功指标剔除，仍报告全程跌倒。速度连续片段、抬脚计数不是自然度或正式验收；不能直接把该诊断的相关噪声叠加到训练PPO而忽略采样概率。


同一冻结11000的生产rollout/GAE诊断：

```powershell
python scripts/amp_tools/run_temporal_probe.py launch --kind advantages --training-state czy/exp1/records/exp1.8_training.json --job czy/exp1/records/exp1.8_evaluation/directional_advantage_job.json
python scripts/amp_tools/run_temporal_probe.py collect --job czy/exp1/records/exp1.8_evaluation/directional_advantage_job.json
```

使用生产collect_rollout连续48次、总计4,718,592采样，不调用policy/discriminator更新；保留正式训练的噪声/RSI/完整指令。核对原始GAE重建与实际全局归一化一致，分命令/速度状态汇总，并在8/24/48轮各抽最多4096样本/组计算条件actor梯度；参数和normalizer逐张量核对不变、新Adam状态始终为空。条件梯度范数与组均值不独立证明优化缺陷，也不替代学习消融。原进程终止且DIRECTIONAL_ADVANTAGE_AUDIT_OK才收集完成。


当前exp1.9正式状态为czy/exp1/records/exp1.9_training.json：真实11000→11250、4096/seed5，96步rollout/250更新等98,304,000新采样预算；预检gpu_preflight_state.json的11005不是正式源。代码、模型输入和warm-start记录已在deployment_verification.json独立核对。最终回放已排队到exp1.9_evaluation/replay_job.json，不要再次启动；查询仍用server_status.py。训练完成后按该状态取回实际11250，再沿用collect_replay_archive.py分阶段收齐严格三件套和六项辅助，不放宽exp1.6验收阈值。


## exp1.10 DH + AMP

x1_amp_exp110恢复66帧历史CNN、5帧短历史和3维缩放机身速度估计；策略外部输入6336、内部actor输入547、动作29。critic保持99维特权输入，隐藏层恢复768/256/128。速度MSE监督只取critic_obs[:,96:99]，系数1，与PPO联合更新；推理不输入真实线速度。原AMP判别器、奖励、PD和50Hz沿用exp1.9。

用户指定：本地只预检，正式云账号limxmts0drnrwf12bo@uberip.com，从零6000轮/4096环境/seed5，96步rollout，共2,359,296,000采样。旧MLP checkpoint不能普通resume/warm_start到DH。训练状态将在实际任务创建后存入czy/exp1/records/exp1.10_training.json；预检汇总只保存一份preflight_summary.json。

play_amp.py --export_policy和export_policy_dh.py均导出完整DH图，不能只导出actor。export_onnx_dh.py沿用环境6336维输入并排除JIT元数据侧车。当前本地70项CPU测试和4096网络批量推理通过；包含跨轮rollout释放回归，避免前一轮9.28GiB观测缓存与下一轮并存。正式GPU运行证据见同一实验状态与笔记。


## exp1.11 DH训练节奏

x1_amp_exp111继承exp1.10，只改24步rollout、2 epochs、4小批；4096/seed5/6000从零，总589824000样本。其余设置包括D16次/轮保留。旧exp1.10任务主动停止，状态和首次OOM记录保留；新状态czy/exp1/records/exp1.11_training.json。本地71项CPU测试通过，回放/导出沿用完整DH路径。


## exp1.12 DH从零课程

x1_amp_exp112：4096/seed5/6000，从零；24步/2 epochs/4小批，D4更新、entropy0.001、std初始0.3/上限0.5。0–200轮五项站立代价权重0.1且AMP速度门控放宽为1，200–1500线性恢复，之后完整要求。回放默认完整课程；checkpoint schema4校验课程与噪声上限，resume按绝对iter继续。状态records/exp1.12_training.json；本地76项测试和4096批完整JIT通过。正式指定账号limxmts0drnrwf12bo@uberip.com，Git新代理10.12.202.95:1080。

## exp1.14 / exp1.14t 两条参考指导路线

正式状态分别为 `czy/exp1/records/exp1.14_training.json` 和 `exp1.14t_training.json`。两者继承同一13t最终6000策略、重置Critic/D/Adam/replay、本轮0→6000；各自分支为 `codex/exp1-14`（98维帧，相位及q/dq/脚端指导）和 `codex/exp1-14t`（96维帧，真实运动相似度减静态基线）。正式账号 `limx2f16enckjjsg@maxxspace.com`，4096/seed5、24步、2×4 PPO、D4；输入输出同一固定PT目录，输入文件名以 `input_` 开头，不能把输入6000误认为新输出。

`preflight_reference_transfer.py --route 114[t] --source <model> --output <json>` 在CPU合成物理环境验证生产网络/真实源模型、4096条分批推理、真实PPO/AMP更新、完整JIT及保存恢复，不代表Isaac Gym验收。`publish_reference_routes.py` 通过临时Git索引和明确白名单生成独立提交，对准确提交树运行回归与真实源预检后普通推送，保留原HEAD和索引。

`launch_reference_route.py` 仅用于本轮已授权任务的首次创建；状态存在或平台同名任务存在即拒绝重建。运行后立即使用 `verify_first_checkpoint.py --state <state>` 捕获启动标记，验证精确 `model_100.pt` 的保存receipt、SDK上传和平台条目；再使用 `inspect_first_reference_checkpoint.py --state <state> --via-server` 在内存核验上传文件hash、实际4096/seed5、输入维度、源模型/控制参数、真实优化器和新D计数，不落盘中间模型。当前两条任务已经创建，后续只监控同一Task-ID，不能再次执行创建脚本。

## exp2.0原速全身模仿

方案与已实施结果见`czy/exp1/exp2.md`。任务`x1_amp_exp20`从exp1.17/model_7000迁移Actor/CNN/状态估计器/log_std，重建Critic、D/normalizer、Adam及replay，新计数0→3000。4096/seed5、24步/2 epochs/4小批，D1；80%请求原速0.5953696966m/s、20%零速，出生缓冲1.5秒，起停各2秒。原生周期1.26秒；AMP移动专家和相位奖励共用带局部接触修正的周期。修正包含关节名，不能按未声明顺序应用数组。

本地只运行CPU预检：

```text
python scripts/amp_tools/preflight_exp20.py --source czy/data/exp1.17/model_7000.pt --output czy/exp1/records/exp2.0_evaluation/preflight_summary.json
```

该工具执行4096条分批推理、4环境fixture中的真实PPO/AMP更新、保存恢复/JIT、生产train入口参数转发、原速参考与过渡几何、实际奖励调用链的10组反例排序。`ready_for_cloud=true`只表示这些CPU门槛通过，不表示GPU物理验收通过。`reference_preflight.png`对比局部修正前后的髋膝和脚底高度。

`prepare_native_reference.py --model <7000源> --output <候选JSON> [--iterations 250] [--initial <已有候选JSON>]`仅生成CPU运动学候选及结果，不能因为优化收敛就替换正式参考。必须检查接触滑移、左右有效离地、关节余量、解析导数及峰速。当前最终运行数据为`humanoid/amp/reference_data/exp20_native_correction.json`和`exp20_sole_hulls.json`；发布、安装、回放部署必须同时包含这两个JSON，不能只打包`.py`。

正式训练的CLI已支持`--warm_start <同一PT输出目录内的input_model_7000.pt> --warm_start_sha256 4edb874efe478baea76af598db86e923c24d62e1653542bf144111a453971ac8 --checkpoint_dir <该输出目录>`，不使用旧`policy_amp_fresh_critic`模式保留D。首100轮继续核验保存receipt、SDK上传与平台模型条目。正式账号、资源和实际提交以任务状态文件为准；当前尚未提交。

CUDA短测可用`humanoid/scripts/smoke_amp.py --task=x1_amp_exp20 ...`，当前只增加支持，没有启动GPU测试。主回放使用`play_amp.py --task=x1_amp_exp20 --vx=0.5953696966 --steps=3000 --reset_mode=standing ...`；源7000在新任务下的策略基线需额外显式`--source_policy --checkpoint_path <model_7000.pt>`，该入口只迁移策略，D新建，不能跨模型比较其style分数。目标checkpoint正常回放不用`--source_policy`。零速辅助使用`--vx=0`；不传vx时为0/原速/0三阶段，可用`--stage_seconds=20`得到60秒起停检查。

`replay_archive.py`与采集器接受`exp2.0`，训练提交快照会纳入参考JSON。`analyze_forward_gait.py`按checkpoint识别NativeCycle，新任务默认6–18、26–38、46–58秒连续窗口；不能用exp1.16/17的奖励重建工具解释新任务。三件套仍只放`czy/data/exp2.0/`，分析和辅助证据放`czy/exp1/records/exp2.0_evaluation/`。

## exp1.17/18绝对相位奖励诊断

`audit_absolute_phase.py --model <checkpoint> --csv <isaac_diag.csv> --output-dir <evaluation目录>` 在CPU校验回放清单中的代码、资源与模型SHA，重现实际参考q/dq、相位时钟、256相位静态基线和有上下限的奖励；输出`reward_mechanism_audit.json`，不启动训练。使用出生/指令/站立缓冲变化后2秒的同口径稳态窗口。支撑分值使用URDF名义质量；平脚均衡支撑仅为奖励反例，不代表物理策略。raw目标取CSV最后物理子步，不等同于子步平均裁剪率。旧`analyze_phase_replay.py`的奖励重建专用于exp1.16，不能将其progress奖励解释成exp1.17/18实际奖励。

## exp1.15 / exp1.15t 正向课程

状态位于 `czy/exp1/records/exp1.15_training.json` 和 `exp1.15t_training.json`，实际账号以状态为准：limx2f1e1sojm8yk@maxxspace.com /4684。两组均先1000次新更新、4096环境、seed5、24步/2×4 PPO/D4，80%请求+0.3m/s、20%零速；AMP策略奖励前500轮为0，500→1500恢复，不能把初期style_reward=0误判为链路故障。t保留同一DH结构与CNN/估计器/噪声，只将exp1.9的Actor精确嵌入短历史核心列；前进教师不代表后退能力。

`preflight_gait_curriculum.py` 验证真实生产模型、静止反例、真实PPO/AMP更新和JIT；t还验证真实云入口的输入准备流程，物理仍为CPU fixture。`publish_gait_curriculum.py` 用临时索引、明确白名单和准确提交预检发布分支。已创建任务不能再次普通执行 `launch_gait_curriculum.py`；`--replace-failed` 只允许终止且日志证明0更新的已诊断教师挂载失败，并保存失败记录，绝不复制运行中的任务。

当前4090D不支持个人存储挂载；官方checkpoint挂载也未实际落文件，因此最终t通过官方checkpoint私有上传通道取得 `checkPointFilePathSignedUrl`，仅在内存准备启动请求。用户已明确允许该私有任务启动参数承载教师签名地址；地址不进入Git、本地状态或笔记。`private_training_input.py` 的checkpoint上传显式设置private ACL并检查匿名403。原失败任务133保留0更新记录，实际训练Task-ID以状态文件为准。

监控优先 `flux_status.py --state <state> --compact`，只覆盖evaluation内一份结构化快照，不新增完整日志。首100仍需 `verify_first_checkpoint.py` 加 `inspect_first_reference_checkpoint.py --via-proxy`，核对实际任务ID、代码、SHA、教师来源及真实优化器。

若Windows原生Flux无法解析平台域名、Windows curl通过指定代理可达，可临时运行 `flux_network_bridge.py`：仅监听127.0.0.1，转发固定官方平台且保留TLS验证，无请求日志。将其打印的本地地址设为当前进程 `FLUX_LOCAL_BASE_URL`；监控结束后关闭自己启动的桥接进程，不永久修改账号或网络配置。

## exp1.16 即时相位修正

`x1_amp_exp116`从实际`czy/data/exp1.15t/model_1000.pt`显式warm-start，新增3000到4000；模型SHA固定在配置中。`policy_amp_fresh_critic`保留Actor/CNN/估计器/log_std、D/normalizer和D Adam动量，重建Critic/PPO Adam及replay；不能用旧`policy_only`代替。前100个新增更新的学习率上限1e-5，保存恢复继续按父iteration1000计算保护期。

只改本轮奖励：有符号逐坐标Huber相位改善、速度基线和软支撑；reference_amplitude/net_displacement权重为0，AMP课程乘数固定0.5。两秒窗口留作诊断；当前相位的前后状态误差比较让静止姿态得分为0，重置和指令切换的一步过渡不领取姿态/脚端改善信用。参考质量日志改为逐坐标倒数Huber质量，不能直接与旧指数分数横向比较。新回放CSV附加奖励使用的reference_phase及reference_q/reference_dq，旧实验CSV不增加列。

本地CPU预检命令：

```powershell
python scripts/amp_tools/preflight_phase_refinement.py --source czy/data/exp1.15t/model_1000.pt --output czy/exp1/records/exp1.16_evaluation/cpu_preflight.json
```

该工具检查真实源模型与AMP动量、4096条分批推理、生产train函数输入准备/新增预算、一次真实PPO/AMP更新、1001保存恢复、JIT及运动学反例；仿真为CPU fixture，不代表物理验收。`publish_phase_refinement.py`已用隔离索引发布准确快照，`launch_phase_refinement.py`已创建TASK_20261030_191；后续读取records/exp1.16_training.json，不得重建任务。新增预算3000、目标计数4000。原`launch_gait_curriculum.py`仍只支持115/115t，不得用该脚本替代116任务。输入与输出共用固定PT目录，首1100用`verify_first_checkpoint.py --state czy/exp1/records/exp1.16_training.json`和`inspect_first_refinement_checkpoint.py`核验；最终4000另行实际交付验证。
