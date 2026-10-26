# AMP 实验工具

这些本机工具从原 `czy/validation/` 迁入，不参与训练算法。以下命令均从项目根目录执行。

| 脚本 | 用途 |
| --- | --- |
| `flux_ops.py` | 调用已安装的Flux CLI，对返回的凭据与签名URL脱敏 |
| `flux_status.py` | 按指定任务状态文件查询训练状态，更新旁边的脱敏日志 |
| `fetch_replay_curves.py` | 获取指定训练任务的AMP曲线，保存到明确指定的附件路径 |
| `replay_archive.py` | 在201.5服务器新目录部署当前代码并启动完整回放 |
| `collect_replay_archive.py` | 取回完成的回放，校验视频/CSV/模型并归档、生成诊断汇总 |
| `analyze_replay.py` | 共用CSV评估：过渡段剔除、速度、姿态、终止及原始/有效目标限位 |
| `analyze_gait.py` | 离线对照参考与策略的同腿周期、关节摆幅、碰撞脚底高度、PD目标与实际运动 |

Flux工具使用本机已安装CLI，继续从已忽略的 `czy/skills/flux-cli/api_key.json` 读取凭据，不将凭据复制到本目录。SSH工具使用Windows自带OpenSSH和密钥登录。

查询已完成训练，状态和日志保存在实验附件目录：

```powershell
python scripts/amp_tools/flux_status.py --state czy/exp1/artifacts/exp_2/training/flux_training.json
python scripts/amp_tools/fetch_replay_curves.py --task-id TASK_20261022_177 --output czy/exp1/artifacts/exp_2/training_curves.json
```

`fetch_replay_curves.py` 支持 `--account`、`--end-time`；`flux_status.py` 根据状态文件中的账号选择凭据。工具不会默认往 `analysis` 写文件。

回放工具读取模型内部iter，按`model_{iter}.pt`打包、校验和归档，支持中期检查点与最终模型；视频仍为最新play的六段60秒、1080p/25fps配置。修改回放配置时需同步其校验条件。以下为后续实验调用示例，执行前须已有用户授权的服务器回放任务：

```powershell
python scripts/amp_tools/replay_archive.py --experiment exp_3 --model czy/data/exp_3_source/model_6000.pt --resource_source /home/robot/czy/exp1/exp_20261026_090135
python scripts/amp_tools/collect_replay_archive.py --experiment exp_3 --model czy/data/exp_3_source/model_6000.pt
```

exp_1.1等独立AMP任务必须显式指定任务与训练来源，可用专用记录目录避免恢复已清理的artifacts目录：

```powershell
python scripts/amp_tools/replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --resource_source /home/robot/czy/exp1/exp_20261026_105426 --task x1_amp_exp11 --training-task-id TASK_20261026_051 --record_dir czy/exp1/exp_1.1_evaluation --extra_seeds 7 11 --check_rsi
python scripts/amp_tools/collect_replay_archive.py --experiment exp_1.1 --model czy/data/exp_1.1/model_6000.pt --record_dir czy/exp1/exp_1.1_evaluation
```

上述部署命令不可对已有任务重复执行。`--primary_only`可在辅助验证期间先取回主三件套，完成后用`--checks_only`补齐同一报告；辅助CSV在临时目录处理后删除，服务器保留原始证据。允许交付目录预先存在同一模型，拒绝覆盖已有回放。笔记依据实际诊断结果更新，不预设“部分达标”。速度稳态窗口剔除指令切换、重置、缓冲结束后2秒；姿态统计剔除回合前100步、终止步和根高度≤0.45m样本。

部署脚本使用系统临时目录打包，取回工具也使用系统临时目录校验，结束后自动清理。已有任务状态或归档时会拒绝覆盖，不要重复启动。

结果位置：

- `czy/data/{实验编号}/`：只存模型、视频、CSV三件套。
- `czy/exp1/artifacts/{实验编号}/`：元数据、日志和摘要。
- `czy/exp1/artifacts/exp_2/initial_validation/`：早期20轮/恢复2轮等验证记录。
- `czy/exp1/artifacts/exp_2/training/`：6000轮正式任务记录和历史资源发布清单。

原 `czy/validation/` 已清空；部署压缩包、短测导出模型、参考工程快照和一次性脚本不保留。

exp_1.1小碎步专项分析（CPU，不启动仿真或修改模型）：

```powershell
python scripts/amp_tools/analyze_gait.py --model czy/data/exp_1.1/model_6000.pt --csv czy/data/exp_1.1/isaac_diag.csv --output_dir czy/exp1/exp_1.1_evaluation
```

该工具目前明确使用exp_1.1六段回放的稳态时间窗与B3连续前进参考；换回放方案时必须调整窗口。输出gait_comparison.json/png；“同腿周期前进距离”不是落脚点步长，最低脚底几何高度不是接触力。

exp_3训练任务为`x1_amp_exp3`。用户已指定从零训练6000轮，启动命令：

```powershell
python humanoid/scripts/train.py --task=x1_amp_exp3 --headless --num_envs=4096 --seed=5 --max_iterations=6000
```

仅训练vx=[-0.2,0.65]，vy/wz固定0；线速度sigma=0.09，角速度sigma=0.25。
PPO学习率初值/最大值1e-4、最小值1e-5，desired_kl=0.02；每小批次更新前计算解析KL(old||current)，超过0.03停止本轮后续更新，末次更新后另计算全rollout KL。学习率每轮调整一次（最大观测KL>0.03则除1.5，<0.01则乘1.5并限制上下界）；一次更新仍可能超限，不是硬KL约束。
AMP风格尺度10，任务/风格混合0.5/0.5；`AMP/style_score`为不乘尺度和dt的分数，回放CSV同名列。原runner完整回合统计及日志保留，追加KL最大值/提前停止/实际更新次数/学习率。
正式任务一次训练6000轮，每100轮保存检查点；原方案的3000+3000已按用户指令改为连续6000轮。runner的max_iterations是本次调用新增轮数，后续若恢复需按实际剩余轮数设置。判别器强度、历史帧数、PD增益、action_scale、LPF及抬脚奖励保持原设置。
回放请显式传`--task x1_amp_exp3`，并为各检查点指定不同experiment和record_dir；不可覆盖正式历史三件套。
