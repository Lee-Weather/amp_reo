---
name: "x1-amp-clean"
description: "维护 X1 29DOF AMP 的 clean 分支约定：humanoid/envs/x1 只保留 x1_amp_config.py、x1_amp_env.py、x1_dh_stand_config.py、x1_dh_stand_env.py 四个文件，只注册 x1_dh_stand / x1_amp / x1_amp_base 三个任务；x1_amp 是 exp2.12 合并后的单文件最终算法，x1_amp_base 是原始 exp1.0 基线。用于修改这份单文件实现、检查目录或注册表是否被重新污染、在 Flux 上验证 x1_amp、以及需要从历史分支恢复 exp1.x/exp2.x 链时。"
---

# X1 AMP clean 版维护规范

## 目标状态

```
humanoid/envs/x1/
  x1_amp_config.py        # X1AMPCfg/PPO（基线） + X1AMPFinalCfg/PPO（exp2.12 合并）
  x1_amp_env.py           # X1AMPEnv（基线） + X1AMPFinalEnv（exp2.12 合并）
  x1_dh_stand_config.py   # 12 自由度 DH 基线
  x1_dh_stand_env.py
```

注册表 [humanoid/envs/\_\_init\_\_.py](../../humanoid/envs/__init__.py) 只允许三个任务：

| task | 环境类 | 配置类 | 说明 |
| --- | --- | --- | --- |
| `x1_dh_stand` | `X1DHStandEnv` | `X1DHStandCfg` | 原 12 自由度基线 |
| `x1_amp` | `X1AMPFinalEnv` | `X1AMPFinalCfg` | exp2.12 合并后的最终算法 |
| `x1_amp_base` | `X1AMPEnv` | `X1AMPCfg` | 原 29 自由度 exp1.0 基线 |

`X1AMPFinalEnv` 和 `X1AMPFinalCfg` 必须**直接继承** `LeggedRobot` / `LeggedRobotCfg`，不允许再挂任何 `x1_amp_exp*` 祖先。

## 关键数值（修改前先核对）

| 项目 | `x1_amp`（Final） | `x1_amp_base`（基线） |
| --- | --- | --- |
| 策略输入 | `frame_stack=66 × num_single_obs=98 = 6468` | `5 × 96 = 480` |
| 价值输入 | 101 | 99 |
| AMP 历史 | `3 × 86` | `3 × 86` |
| 判别器风格缩放 | 10 | 2 |
| runner | `Exp25AMPOnPolicyRunner` | `AMPOnPolicyRunner` |
| policy | `ActorCriticDHAMP`（含 long_history + 状态估计器） | 默认 MLP |
| native 速度 | `0.5953696966` | 指令区间[-0.35, 0.65] |
| 日志身份 | `experiment_name=x1_amp`, `run_name=amp_final` | `experiment_name=x1_amp_base` |

## 硬约束（不要破坏）

1. **不要重新引入 `x1_amp_exp*.py`**，也不要在注册表里恢复 exp 任务。历史实验代码在分支 `codex/merge-exp212-flat`，实验记录在 `czy/exp1/exp1.md`、`exp1_part2.md`、`exp1_part3.md`、`exp2.md`。
2. **不要删除 `humanoid/amp/` 下的课程与参考模块**。`X1AMPFinalEnv` 仍依赖：
   `exp20_commands.py`、`exp20_rewards.py`、`exp23_curriculum.py`、`exp23_runner.py`、
   `exp24_curriculum.py`、`exp24_runner.py`、`exp25_curriculum.py`、`exp25_runner.py`、
   `exp26_curriculum.py`、`exp26_swing.py`、`exp27_balance.py`、`exp27_curriculum.py`、
   `exp28_curriculum.py`、`exp28_tracking.py`、`exp29_curriculum.py`、`exp29_phase.py`、
   `contact_domain.py`、`nominal_startup.py`、`native_reference.py`、`gait_curriculum.py`、
   `gait_rewards.py`、`anchored_runner.py`、`conservative_runner.py` 等。
3. **不要删除 `humanoid/amp/reference_data/`**：`exp20_native_correction.json`、`exp20_sole_hulls.json`、`exp22_policy_state_bank.json` 是运行期必需输入。
4. **不要硬编码观测维度**。脚本应从 `cfg.env.num_observations` / `cfg.env.num_privileged_obs` 读取；`validate_amp.py` 已经这样处理。
5. **不要在 fresh smoke 里使用 `Exp25AMPOnPolicyRunner`**：它的 `learn()` 要求已验证的续训 origin。fresh smoke 必须走 `smoke_amp.py` 里的 `AMPOnPolicyRunner` 分支。
6. **不要把 12 自由度 DH 与 29 自由度 AMP 的模型/配置混用**；两者 schema、控制维度和 checkpoint 不兼容。

## 修改流程

1. 只改 `x1_amp_config.py` / `x1_amp_env.py` 里的 Final 类；不要新增实验子类。
2. 奖励或观测改动必须同时检查配置路径：所有 `self.cfg.<path>` 都要能在 `X1AMPFinalCfg` 解析。
3. 训练参数只改 `X1AMPFinalCfgPPO`，不要改基线类（基线是 exp 链的根，虽然本分支已无 exp 任务，但仍作为 `x1_amp_base` 使用）。
4. 本地跑检查器与测试：

```bash
python czy/skills/x1-amp-clean/check_clean.py
python -m unittest discover -s tests -v
```

5. 改动提交到 `clean` 分支；不要 force push。GitHub 直连失败时使用代理：

```bash
git -c http.proxy=socks5h://10.12.201.122:39009 -c http.sslBackend=openssl push origin clean
```

## 验证协议

### 1. 静态检查（必需）

```bash
python czy/skills/x1-amp-clean/check_clean.py
```

检查项：四个环境文件、三个注册任务、无 exp 模块导入、必需 amp 模块与 reference_data、config 解析后的 6468/101/480/99、Final 类无实验继承、测试数量与导入。

### 2. GPU 环境验证（Flux）

用 `flux-cli` skill 建任务，启动脚本：

```bash
gm-run amp_reo/humanoid/scripts/validate_amp.py --task=x1_amp --num_envs=8 --headless
```

期望 stdout：

```
SIM_FK_MAX_ERROR_M=< 0.002
AMP_SIM_VALIDATION_OK dofs=29 obs=6468 critic=101 amp=3x86
EXP20_REFERENCE_AUDIT passed=true
CONTACT_RANDOMIZATION_READY verified=true
NOMINAL_STARTUP_READY verified=true
NOMINAL_STARTUP_OBS_VERIFIED verified=true
```

任何一项缺失或 `obs`/`critic` 维度不符，先不要进入训练。

### 3. 训练链路 smoke

```bash
gm-run amp_reo/humanoid/scripts/smoke_amp.py --task=x1_amp --num_envs=256 --headless \
  --smoke_iterations=1 --resume_iterations=1 --output_dir=/workspace/amp_reo/logs/flat_smoke
```

期望最后一行：

```
GPU_SMOKE_OK ... status=passed checkpoint_restore_exact=true inference_shape=[256, 29]
observation_shape=[256, 6468] state_estimator_supervision_index=98
```

覆盖 PPO/判别器更新、checkpoint 精确恢复、env learning-state JSON 往返和推理维度。

### 4. 与历史版本做等价对比

需要确认“合并没有改变行为”时，从历史分支取 exp2.12 配置做解析对比：

```bash
git show origin/codex/merge-exp212-flat:humanoid/envs/x1/x1_amp_exp212_config.py > /tmp/exp212_config.py
# 用 Python 逐叶子键比较（resolved config dict）
```

允许的差异只有 `runner.experiment_name` 和 `runner.run_name`。

## Flux 任务模板（validate）

```json
{
  "taskBaseInfo": {
    "projectId": "<PROJECT_ID>",
    "taskType": "1",
    "trainType": "1",
    "taskName": "x1-amp-clean-validate",
    "taskDescription": "Validate clean-branch x1_amp: package import, obs 6468, critic 101, auto-reset boundary",
    "taskTag": ["verify", "clean"],
    "goodsId": "ESKU000001",
    "imageId": "BJX00000001",
    "imageVersion": "V000412",
    "personalDataPath": ""
  },
  "taskCodeInfo": {
    "codeType": "2",
    "codeUrl": "[{\"codeUrl\":\"https://github.com/Lee-Weather/amp_reo.git\",\"versionType\":\"1\",\"versionName\":\"clean\"}]",
    "mainCodeUri": "amp_reo/humanoid/scripts/validate_amp.py",
    "hparamsPath": "amp_reo/humanoid/envs/x1/x1_amp_config.py",
    "startScript": "gm-run amp_reo/humanoid/scripts/validate_amp.py --task=x1_amp --num_envs=8 --headless",
    "isOpen": "0"
  },
  "runtimeReminderConfig": {"enableRuntimeReminder": false, "reminderDurations": []}
}
```

创建前先 `--dry-run`（exit 10），再正式创建、`--yes task run`、`flux task logs --follow --raw --no-request-log`。

## 禁止事项

- 在 `humanoid/envs/__init__.py` 里重新 import `x1_amp_exp*`。
- 给 `X1AMPFinalEnv` / `X1AMPFinalCfg` 增加实验父类。
- 把 `x1_amp` 指回 `X1AMPEnv`（那会让最新算法退回 exp1.0）。
- 删除 `humanoid/amp/reference_data/` 或课程/参考模块。
- 在 fresh smoke 中直接实例化 `Exp25AMPOnPolicyRunner` 并调用 `learn()`。
- 用 `x1_amp_exp*` 任务名启动（本分支已不存在）。
- 在实验记录目录 `czy/exp1/records` 中重新堆积 evaluation 产物（该目录已从 clean 分支移除；大文件请放仓库外）。

## 历史与恢复

| 内容 | 位置 |
| --- | --- |
| exp1.x/exp2.x 完整代码与测试 | 分支 `codex/merge-exp212-flat` |
| 合并前完整工作区快照 | 提交 `85f0f77`（父 `34d8c09`） |
| 清理删除的 179 个代码文件 | `E:\X1\amp_re0\deleted_files_20261104.tgz` |
| exp1 实验 records 归档 | `E:\X1\amp_re0\czy_exp1_records_20261105.tgz` |
| 实验叙述记录 | `czy/exp1/exp1.md`、`exp1_part2.md`、`exp1_part3.md`、`exp2.md` |

恢复单个文件：

```bash
git show origin/codex/merge-exp212-flat:humanoid/envs/x1/x1_amp_exp212_env.py > /tmp/x1_amp_exp212_env.py
```

恢复整条链：

```bash
git switch -c restore-exp origin/codex/merge-exp212-flat
```

## 验收清单

- [ ] `humanoid/envs/x1` 只有四个 `.py`
- [ ] 注册表只有 `x1_dh_stand`、`x1_amp`、`x1_amp_base`
- [ ] `check_clean.py` 全部 PASS
- [ ] 本地 11 个测试通过
- [ ] Flux `validate_amp.py --task=x1_amp` 通过
- [ ] 需要训练链路时 `smoke_amp.py` 返回 `GPU_SMOKE_OK`
- [ ] 提交已推送到 `origin/clean`，没有 force push