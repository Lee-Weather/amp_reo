# 私有模型提交云训练：exp1.15t 实证流程

适用于本仓库通过 Git 提交代码、另外提供本地私有 checkpoint / 教师模型的 Isaac Gym 训练。重点解决“文件已上传却无法挂载”“训练运行但没有输出模型”。普通无外部模型的任务仍按主 skill 提交，不必走额外上传流程。

本章依据 2026-10-30 的启动和首模验收记录；不代表此后任务状态或最终步态效果。账号、预算、镜像版本和模型来源均为案例值，后续按用户要求及平台实查选择。本文不授权新建训练、公开模型、扩大预算或切换到其他账号。

## 1. 已验证的成功路径

**代码走 Git，私有模型走官方 checkpoint 上传通道；从 task info 取得签名下载地址，训练时下载并校验 SHA，不依赖模型自动挂载。**

```text
本地模型 → 官方上传策略 → private ACL 的 OSS 对象
                               ↓
taskCodeInfo.checkPointFilePathSignedUrl（仅在内存使用）
                               ↓
私有训练任务：trainType=1、personalDataPath为空、gm-run
                               ↓
教师模型下载到内存 + 主源模型准备到固定 PT 目录
                               ↓
真实迭代 → 首模保存 → SDK上传 → 平台条目 → 下载内容验收
```

本次有两个不同用途的输入，不能混淆：

| 输入 | 来源 | 云端处理 |
| --- | --- | --- |
| 稳定 DH 基础 | exp1.14 的 `model_5300.pt` | 由源任务 `model list` 返回的 `policUrlDown` 下载；保留指定策略组件，重建 Critic / AMP / 优化器 |
| 前进教师 | 本地 exp1.9 的 `model_11250.pt` | 官方 checkpoint 私有上传；签名下载到内存，校验后精确迁移 Actor |
| 本轮输出 | `model_100.pt` … `model_1000.pt` | 本轮从 0 计数，保存与上传到新任务的固定输出目录 |

`Learning iteration 0/1000` 在这里是新训练计数，不是丢失了基础策略或教师。应检查迁移标记和 checkpoint 内的 `training_origin`。

## 2. 提交前先确认身份、输入和代码

1. 显式选取用户指定账号，调用 `auth whoami` 核对邮箱与 userId，再核对项目归属。项目 helper `flux_ops.py` 的默认账号可能属于旧实验；每个进程都应从当前计划/状态设置 `flux_ops.ACCOUNT`，不能依赖上一次调用的 Python 全局变量。
2. 查询当前 resource / official image / image versions，确认资源可用及其挂载能力。继承模型不意味着一定要用平台 `trainType=2`；平台输入方式与算法的 resume / policy transfer 分开判断。
3. 明确两个模型的实际文件、SHA、关节顺序、观测约定、控制周期、PD 和动作尺度。CPU 预检检查真实 checkpoint、生产网络、迁移、更新和导出；合成物理不能替代 GPU 训练/步态验收。
4. 发布预检过的准确代码提交，核对 `ls-remote` 与目标 SHA 一致。逐一用 `git cat-file -e <commit>:<仓库相对路径>` 确认mainCodeUri、hparamsPath及新增入口依赖确实在该提交中，并在准确归档中编译入口；本地文件存在或本地测试通过不能代替发布文件检查。脏工作区使用明确文件白名单和隔离索引，保留用户现有分支、索引和无关改动。Git 流程见 [git-mirror.md](git-mirror.md)。
5. 查询目标项目的同名任务并读取本地状态。创建超时先查实际任务是否已生成；成功取得 Task-ID 后先写状态，再启动，避免重复创建计费任务。

## 3. 本地教师模型的私有上传

### 3.1 使用 checkpoint 上传通道

flux-cli 当前没有本地文件上传子命令。本次项目扩展复用了官方网页的接口：

```text
GET https://internal.limxdynamics.com/dev-api/api/common/upload-directly?fileName=...
X-Api-Key: 从选定账号凭据在内存读取
```

返回 `host`、`key`、`policy`、`signature`、`security_token` 和 OSS V4 签名字段。按该响应向 `host` 发 multipart POST，文件字段最后提交；上传表单包括：

```text
success_action_status=200
policy
x-oss-signature
x-oss-signature-version
x-oss-credential
x-oss-date
key
x-oss-security-token
x-oss-object-acl=private
callback（响应非空时）
file
```

先校验本地 SHA；上传后检查不带签名的对象 GET 返回 403。取得有效签名地址后再下载，核对字节数和 SHA，才算输入内容验证完成。对象 key、文件名、大小、SHA 和私有权限验证结果可以留存；策略、临时 token、签名和下载地址不进入长期记录。

实现见 [private_training_input.py](../../../../scripts/amp_tools/private_training_input.py) 的 `upload_checkpoint()`。注意：该脚本直接执行的 CLI 入口上传的是 **个人存储 TOS**，不是本节的 OSS checkpoint 通道；不要把两者混用。当前函数的逻辑文件名仍带 exp1.15t / 11250，复用到新实验时应调整命名和 receipt 路径。

### 3.2 从官方任务字段取得下载地址

对已关联该私有对象的、用户有权访问的任务读取：

```python
info = flux.call("task", "info", "--task-id", input_task_id)["data"]
teacher_url = info["taskCodeInfo"]["checkPointFilePathSignedUrl"]
# 不打印 info 或 teacher_url；下载到内存并验证预期 SHA。
```

本次已验证该字段由失败任务 `TASK_20261030_133` 返回，所指对象与本地教师模型 SHA 一致。不要猜测对象 URL、复制历史签名，或用账号 API Key 代替模型下载地址传给训练脚本。

后续准备新输入时，可先在未运行草稿中关联官方上传返回的对象 key：`taskCodeInfo.checkPointFilePath`，再读取 task info 检查签名字段。**本次没有单独验证“从未启动的草稿一定返回签名地址”**；若字段为空，先检查当前接口或其他已授权的输入来源，不为获取地址而启动 GPU，不故意制造失败任务。

### 3.3 签名地址的授权与留存

签名 URL 是访问凭据；Base64 只是传参编码，不是加密。本次用户明确批准把教师签名地址用于该账号下的私有对照训练启动参数，最终任务 `isOpen=0`。地址有效期应从当次响应检查，不能假定只有几分钟；本次曾返回约 7 天。

已有授权覆盖同一用途时沿用，不反复询问；此案例不自动授权公开任务、公开模型、其他文件或 API Key 的传递。若自动审批拒绝相同操作，不换工具或间接执行绕过；说明拦截动作与凭据留存风险，取得所需批准，或采用已验证且风险更低的输入通道。

完整 `startScript` 只在请求构造中使用。一次性请求 JSON 可能短暂包含签名参数，用临时目录在成功/失败后清理；长期状态和笔记仅保留脱敏命令，不能宣称地址从未经过本地临时文件。

## 4. 构造最终训练任务

适用于本次已确认挂载不可靠的资源组合：

| 字段 | 最终做法 |
| --- | --- |
| `taskBaseInfo.trainType` | `"1"`，运行时准备输入 |
| `taskBaseInfo.personalDataPath` | 空字符串 |
| `taskCodeInfo.codeType` | `"2"`，Git |
| `taskCodeInfo.isOpen` | `"0"`，私有任务 |
| `mainCodeUri` / `hparamsPath` | 含 clone 后的 `amp_reo/` 前缀，指向实际入口与配置 |
| `startScript` | 以 `gm-run` 开头，输入下载参数只在内存组装 |

本次脱敏命令如下；**省略了必需的私有输入参数，不能直接作为可运行命令复制**：

```text
gm-run amp_reo/humanoid/scripts/train_guided_amp.py --task=x1_amp_exp115t --headless --num_envs=4096 --seed=5 --max_iterations=1000 --run_name=exp1.15t --checkpoint_dir=/workspace/amp_reo/logs/x1_amp_exp115t/exported_data/exp1.15t_runtime --warm_start=/workspace/amp_reo/logs/x1_amp_exp115t/exported_data/exp1.15t_runtime/input_exp114_model_5300.pt --warm_start_sha256=5c7df155c40ea87f2bc3a23301f193d15fb0cb511b21961d390252d6af28e1d5
```

请求构造时附加 `--warm_start_url_b64` 和 `--teacher_url_b64`；不要把实际值写进文档、聊天、普通日志或 Git。教师预期 SHA 来自预检过的配置，入口下载后校验，在 `BytesIO` 中加载。签名地址过期时重新从授权来源取得，不能沿用旧启动参数盲重试。

若草稿需要调整，先取全量 task info，保留完整 `taskBaseInfo`、`taskCodeInfo`、`runtimeReminderConfig`，修改目标字段并保留 `taskCodeInfo.taskId`；不要用省略字段的部分 payload 覆盖 RL 任务配置。移除旧挂载方式遗留的 checkpoint / resume 字段，核对私有性和资源不被改掉。

创建/编辑/运行均先 dry-run，捕获而不打印含敏感参数的完整预览。正常启动后核对实际 userId、commit、真实迭代；“提交成功”或平台状态 3 本身不足以说明训练脚本已开始更新。

## 5. 防止模型训练完却没有上传

本项目曾观察到 SDK 只跟踪首次识别的 PT 目录。采用一个新的、持久存在的目录容纳基础模型输入和训练输出：

```text
/workspace/amp_reo/logs/x1_amp_exp115t/exported_data/exp1.15t_runtime/
  input_exp114_model_5300.pt
  model_100.pt
  model_200.pt
  ...
  model_1000.pt
```

- 主源输入在该目录准备；临时下载用 `.part`，校验后移到最终路径，再按现有 helper 触发输入文件事件。
- 教师在内存读取，不在另一个临时目录写出 `.pt`，避免 SDK 锁定临时目录。
- 输出路径与 `--checkpoint_dir` 相同。拒绝覆盖已有训练结果目录；如确有额外共目录输入，须逐一验证 SHA，而不是放宽到接受任意旧文件。
- 输入名以 `input_` 区分。平台里看到 5300 / 11250 不表示本轮已训练到该轮数；只验收当前任务精确的 `model_100.pt`，最终再验收 `model_1000.pt`，不能取列表里的最大数字冒充新输出。

相关实现：[checkpoint.py](../../../../humanoid/amp/checkpoint.py)、[train_guided_amp.py](../../../../humanoid/scripts/train_guided_amp.py)。

## 6. 本次失败点与停止重试的条件

| 现象 | 证据/原因 | 后续做法 |
| --- | --- | --- |
| 私人存储文件上传成功，task run 被拒 | 601：当前 4090D 商品不支持个人存储挂载 | 使用其他已授权输入通道；不要继续修改个人存储路径反复 run |
| 官方 checkpoint 输入已配置，Python 仍报文件不存在 | 133 实际 0 更新，159 秒后结束；SDK 未将教师文件落到指定位置 | 使用已验证的签名下载；不把配置存在当成文件已存在 |
| 失败后 task edit 被拒 | 1002128：`Only draft tasks can be edited` | 仅状态 0 草稿可编辑；不要重复编辑终态任务 |
| Flux 域名解析失败，或其代理连接失败 | Windows curl 经指定代理可达 | 按第 7 节验证并采用临时本机转发，不关闭 TLS 验证 |
| `CHECKPOINT_SAVED` 存在，model list 没有输出 | 只证明本地保存，可能未上传或目录未被跟踪 | 检查 SDK 队列/上传成功及正确 PT 目录；不自动重跑长训 |

替换失败任务前，确认旧任务终止、没有实际学习更新和有效训练输出，且用户授权仍覆盖账号、用途及预算；保存旧 Task-ID、失败原因、时间和 0 更新证据。创建超时、排队或暂时无日志不等于失败，不能据此复制任务。有实际更新时先检查已保存模型，按授权处理恢复和剩余预算。

本次终态 133 不可编辑，因此保留失败记录，建立私有替代任务 143，训练预算仍为 1000 次新更新。不能把“替换零更新失败任务”推广成遇到任意状态 6 都自动重开。

## 7. Windows 网络替代方案

只在 Flux 直连失败且 Windows curl 经用户指定代理已实测能访问官方 API 时使用 [flux_network_bridge.py](../../../../scripts/amp_tools/flux_network_bridge.py)：仅监听 `127.0.0.1` 的临时端口，把请求转发到固定官方服务，保留上游 HTTPS 证书验证，不记录请求头或正文。

运行桥接工具后，将它打印的完整地址设置为当前会话 `FLUX_LOCAL_BASE_URL`，供本仓库 helper 使用；直接调用官方 flux 时，使用其支持的 `FLUX_BASE_URL`。`FLUX_LOCAL_BASE_URL` 是项目自定义变量，不是官方 CLI 参数。代理与端口按当次实际情况选择，不照抄旧的 12826 端口。

记录自己启动的桥接进程 PID，使用完关闭该进程。只修改当前进程环境，不永久改变账号、全局代理或 TLS 配置；本机转发也不改变创建任务和传递凭据所需的授权。

## 8. 启动通知和首模交付验收

确认真实迭代后，可以先通知用户已进入训练，并说明首模尚待验证；仅“提交成功”不能这样报告。完成下列证据后，再报告首模交付链路已验证，证据须属于同一实际任务和准确代码提交：

1. `CHECKPOINT_INPUT_READY`：主源路径、字节数、SHA 正确。
2. `POLICY_TRANSFER_OK`：来源 5300、本轮 0、保留与重建组件符合计划。
3. `TEACHER_TRANSFER_OK`：实际教师 SHA、来源 11250、迁移方法正确。
4. 真实 `Learning iteration` 推进，日志无启动异常；AMP 早期奖励为 0 在本次前 500 轮课程中是预期行为。
5. `CHECKPOINT_SAVED` 给出新 `model_100.pt` 的大小和 SHA。
6. SDK 明确 `uploaded successfully`，平台 model list 出现当前任务精确的 `model_100.pt` / checkpoint 100。仅 `New file detected globally` 不算上传成功。
7. 下载该输出到内存，SHA 与保存 receipt 一致；检查真实配置、模型来源、全部参数/优化器有限及完整 DH JIT。

现有只读工具（从仓库根目录执行）：

```powershell
python scripts/amp_tools/verify_first_checkpoint.py --state czy/exp1/records/exp1.15t_training.json --timeout-seconds 1800
python scripts/amp_tools/inspect_first_reference_checkpoint.py --state czy/exp1/records/exp1.15t_training.json --via-proxy
python scripts/amp_tools/flux_status.py --state czy/exp1/records/exp1.15t_training.json --compact
```

以上针对本次已有任务，不能据此创建或重开训练。验收 helper 的特殊路径也应核对适用条件；例如首模检查针对当前配置的 100 轮，不能给其他更新节奏机械套用相同计数。

本次首模实测：6468→29、4096 环境、seed5、24步 rollout、PPO 2 epochs×4小批、D4；PPO Adam 步 800、D Adam 步 400，normalizer 计数 `100×4×1024×2×3=2457600`，所有参数/优化器有限，完整 JIT 一致。PPO 开启 KL 提前停止时实际步数可少于 800，不能通用地要求等于 800。

只覆盖 evaluation 中一份结构化进展快照，避免堆积一次性全量日志。更新实验索引和状态中的实际 Task-ID，明确首模验收不是训练收敛、训满 1000 或步态达标。

## 9. 案例证据与脚本复用边界

| 项目 | 本次已验证值 |
| --- | --- |
| 账号 / userId / 项目 | `limx2f1e1sojm8yk@maxxspace.com` /4684 /`PRO_20261030_009` |
| 正式对照任务 | `TASK_20261030_143`，2026-10-30 15:36:13 开始 |
| 失败任务 | `TASK_20261030_133`，0 更新；不作为正式训练结果 |
| Git 分支 / 提交 | `codex/exp1-15t` /`2efee2451244a06dd88aec1b22a9c2ae7318ade6` |
| 算力 / 镜像 | `ESKU000001`，1×4090D24G /`BJX00000001` `V000412`，仅为当日实查值 |
| 首阶段预算 | 1000 新更新；4096 /seed5；98,304,000 样本 |
| 首模 | `model_100.pt`，15013050 字节，平台模型 ID 4229788 |
| 首模验证时刻 | 2026-10-30 15:42:10 +08:00，日志已到 106 轮 |

原始证据：[训练状态](../../../exp1/records/exp1.15t_training.json)、[私有输入 receipt](../../../exp1/records/exp1.15t_evaluation/teacher_checkpoint_input.json)、[失败记录](../../../exp1/records/exp1.15t_evaluation/failed_startup.json)、[首模上传证据](../../../exp1/records/exp1.15t_evaluation/first_checkpoint_verification.json)、[首模内容检查](../../../exp1/records/exp1.15t_evaluation/first_checkpoint_inspection.json)。

项目脚本位于仓库 `scripts/amp_tools/`，单独复制 skill 文件夹不会自动带走这些工具：

| 工具 | 用途及边界 |
| --- | --- |
| [preflight_gait_curriculum.py](../../../../scripts/amp_tools/preflight_gait_curriculum.py) | 本次 115 /115t 生产模型与 CPU 预检，不是通用物理验收 |
| [publish_gait_curriculum.py](../../../../scripts/amp_tools/publish_gait_curriculum.py) | 本次分支的隔离索引发布；存在 publication 时不重复运行 |
| [launch_gait_curriculum.py](../../../../scripts/amp_tools/launch_gait_curriculum.py) | 保存了本次成功替换请求的构造，但账号/模型/预算写死；普通 115t 路径仍是失败的个人存储路径，不能照跑。`--replace-failed` 也不是新实验的默认提交方式 |
| [verify_first_checkpoint.py](../../../../scripts/amp_tools/verify_first_checkpoint.py) / [inspect_first_reference_checkpoint.py](../../../../scripts/amp_tools/inspect_first_reference_checkpoint.py) | 首模保存/上传/下载内容验收，核对任务 ID 和 commit |
| [flux_status.py](../../../../scripts/amp_tools/flux_status.py) | `--compact` 精简状态快照 |

后续新实验应按本文最终输入方式适配提交器，直接接受已验证、已授权的教师下载来源，移除个人存储挂载依赖；不能为了复用旧脚本而先运行一个必然失败的任务。复用时参数化账号、源任务、模型/SHA、实验编号、预算、分支与镜像，保留身份核对、防重复和首模验收。
