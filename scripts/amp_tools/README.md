# AMP 实验工具

这些本机工具从原 `czy/validation/` 迁入，不参与训练算法。以下命令均从项目根目录执行。

| 脚本 | 用途 |
| --- | --- |
| `flux_ops.py` | 调用已安装的Flux CLI，对返回的凭据与签名URL脱敏 |
| `flux_status.py` | 按指定任务状态文件查询训练状态，更新旁边的脱敏日志 |
| `fetch_replay_curves.py` | 获取指定训练任务的AMP曲线，保存到明确指定的附件路径 |
| `replay_archive.py` | 在201.5服务器新目录部署当前代码并启动完整回放 |
| `collect_replay_archive.py` | 取回完成的回放，校验视频/CSV/模型并归档、更新笔记 |

Flux工具使用本机已安装CLI，继续从已忽略的 `czy/skills/flux-cli/api_key.json` 读取凭据，不将凭据复制到本目录。SSH工具使用Windows自带OpenSSH和密钥登录。

查询已完成训练，状态和日志保存在实验附件目录：

```powershell
python scripts/amp_tools/flux_status.py --state czy/exp1/artifacts/exp_2/training/flux_training.json
python scripts/amp_tools/fetch_replay_curves.py --task-id TASK_20261022_177 --output czy/exp1/artifacts/exp_2/training_curves.json
```

`fetch_replay_curves.py` 支持 `--account`、`--end-time`；`flux_status.py` 根据状态文件中的账号选择凭据。工具不会默认往 `analysis` 写文件。

回放工具目前对应 `model_6000.pt` 和最新play的六段60秒、1080p/25fps配置；修改训练轮数或回放配置时需同步其校验条件。以下为后续实验调用示例，执行前须已有用户授权的服务器回放任务：

```powershell
python scripts/amp_tools/replay_archive.py --experiment exp_3 --model czy/data/exp_3_source/model_6000.pt --resource_source /home/robot/czy/exp1/exp_20261026_090135
python scripts/amp_tools/collect_replay_archive.py --experiment exp_3 --model czy/data/exp_3_source/model_6000.pt
```

部署脚本使用系统临时目录打包，取回工具也使用系统临时目录校验，结束后自动清理。已有任务状态或归档时会拒绝覆盖，不要重复启动。

结果位置：

- `czy/data/{实验编号}/`：只存模型、视频、CSV三件套。
- `czy/exp1/artifacts/{实验编号}/`：元数据、日志和摘要。
- `czy/exp1/artifacts/exp_2/initial_validation/`：早期20轮/恢复2轮等验证记录。
- `czy/exp1/artifacts/exp_2/training/`：6000轮正式任务记录和历史资源发布清单。

原 `czy/validation/` 已清空；部署压缩包、短测导出模型、参考工程快照和一次性脚本不保留。
