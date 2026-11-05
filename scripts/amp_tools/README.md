# AMP 工具

本分支只保留与 `x1_amp`、`x1_amp_base` 和 `x1_dh_stand` 直接相关的通用工具。

## 云端与产物

- `flux_ops.py` / `flux_status.py` / `flux_network_bridge.py`：Flux 任务查询与网络桥接。
- `fetch_server_checkpoint.py` / `fetch_server_curves.py` / `fetch_amp_checkpoint.py` / `fetch_final_amp.py` / `fetch_replay_curves.py`：checkpoint 与训练曲线下载。
- `server_status.py` / `queue_server_replay.py` / `watch_amp_finals.py`：任务状态与收尾监控。
- `private_training_input.py`：私有 checkpoint 的运行时下载封装。

## 回放与分析

- `replay_archive.py` / `replay_dependency.py` / `collect_replay_archive.py`：回放归档与依赖检查。
- `analyze_replay.py` / `analyze_gait.py` / `analyze_standing.py` / `analyze_forward_gait.py` / `analyze_command_reference.py` / `analyze_amp_features.py` / `analyze_phase_replay.py` / `analyze_scuff_kinematics.py` / `summarize_amp_acceptance.py`：步态、指令与回放分析。
- `cpu_compare.py` / `native_physics.py` / `prepare_native_reference.py` / `build_exp22_bank.py`：参考数据与 CPU 物理对照。
- `mujoco_diagnostics.py` / `sim2sim_amp.py` / `sim2sim_amp_native.py`：MuJoCo sim2sim 诊断。
- `collect_training_curves.py` / `verify_first_checkpoint.py` / `verify_gym_velocity_reference.py`：曲线汇总与首模/速度参考校验。

历史 exp1.x/exp2.x 的 launch、publish、inspect、preflight 工具已从本分支移除；完整版本见分支 `codex/merge-exp212-flat`。实验记录见 `czy/exp1/`。
