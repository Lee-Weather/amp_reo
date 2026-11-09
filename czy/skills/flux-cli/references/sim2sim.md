# 当前29自由度AMP的MuJoCo sim2sim

用户指定后续默认以sim2sim替代Isaac回放。此规则用于当前项目，不授权新训练、修改任务或新增种子。仅seed=5；训练未结束时选定已上传模型并明确为中途评估，训练继续运行。

## 模型与代码

按实验状态文件显式选择账号，核验任务userId和训练commit；从官方model list取得准确输出模型，排除input模型。保存回执可见时比对字节数和SHA；若日志尾部不含回执，如实记录限制，并核验官方HTTPS下载长度、可用对象ETag及checkpoint任务/轮次/来源。ETag的十六进制大小写无关。保存模型到 `czy/data/<实验编号>/model_<轮次>.pt`。

从准确训练commit用 `git -c core.autocrlf=false archive` 提取humanoid和resources到临时目录，避免Windows换行破坏参考SHA。使用本地 `scripts/amp_tools/sim2sim_amp_native.py`，不套用其他项目的观测或PD接口：

```text
python scripts/amp_tools/sim2sim_amp_native.py --root <准确训练快照> --checkpoint <data中的模型> --mjcf-source czy/robots/x1/mjcf --meshes resources/robots/x1_29/meshes --out-dir czy/exp1/records/<实验编号>_evaluation/sim2sim_mujoco_<轮次> --training-commit <commit> --runtime-properties <已验证runtime.json> --passive-probe <已验证passive_probe.json> --seconds 60
```

本地已有MuJoCo、torch和cv2的Python可用；读取checkpoint前设torch单线程，避免CPU开销。临时快照须保留至回放子进程结束。

## 运行时物理对齐（2026-11-04修正，必须执行）

exp2.9/model_13500的准确代码在201.5单环境实测：PhysX将60个URDF惯性link合并为30个刚体；AssetOptions.armature=0.01添加到每个存续刚体（包括浮动基座）的惯量对角线，DOF armature实际为0。不能把AssetOptions.armature直接赋给MuJoCo dof_armature，两个定义不同。DOF damping实际为1；当前EFFORT模式的无接触重复被动试验确认它实际生效，不能在MuJoCo一律清零。

用 `scripts/amp_tools/export_native_runtime.py` 从准确训练代码/资源和checkpoint导出运行时属性及必要状态；`--passive-probe-only`用于3组各10步无接触阻尼核验，不训练策略。CPU工具 `audit_native_transfer.py` 对齐固定关节合并后的完整质量/COM/惯量，并重建当前NativeCycle/B3与quiet专家。首个可追溯实例位于 `czy/exp1/records/exp2.9_long_evaluation/runtime_audit/`。复用导出时必须核对资产、仿真配置、关节顺序与PD一致；这些发生变化需重新导出，不能套用其他机器人的数据。

native回放通过 `native_physics.align_runtime` 验证全张量差异确为link-space armature后，每个存续刚体只加一次惯量，不给固定附属件重复加；DOF armature/damping/friction按验证值映射。若差异不符合已验证模式则拒绝回放，不能放宽断言掩盖不一致。接触求解器/材料组合仍有差异，不能由刚体对齐宣称两引擎完全等价。

2026-11-04之前的回放保留为旧物理映射基线，不能将历史早摔全部归因于策略。相同13500模型的修正回放为 `czy/data/exp2.9/sim2sim_13500_aligned.mp4/.csv`：4.48→5.36秒仍摔，不宣称已通过。新旧视频/CSV必须保留独立名称；已有默认文件时用 `--video-path`、`--csv-path`指定不覆盖的新文件。

## 固定评估条件

沿用当前已验证名义条件：摩擦0.8、原生0.59537m/s命令、1.5秒站姿缓冲+2秒斜坡、无噪声/推扰/重置。脚本从checkpoint读取6468=66×98输入、29输出、关节顺序、原PD、动作尺度、力矩限制、50Hz策略/500Hz物理及参考相位。角速度为MuJoCo局部angular前三项；PD每物理子步计算，无LPF。FK、质量/惯量及JIT输出检查必须通过。首次base z<0.35或upright<0.5记跌倒，继续录3秒后退出；60秒是目标时长，不把提前跌倒说成60秒通过。无需额外GPU短训或默认多次调参重跑。

## 文件和判读

脚本默认将 `sim2sim_<轮次>.mp4` 和 `sim2sim_<轮次>.csv` 写到checkpoint旁，即 `czy/data/<实验编号>/`；同名文件已存在则拒绝覆盖。summary、actor导出、生成MJCF和分析/校验记录保留在records目录。视频全帧解码，校验分辨率/帧数/fps；CSV按50Hz写出，核验所有数值有限。记录模型和视频/CSV的SHA、代码commit、实际时长、跌倒时间及路径。

CSV包含指令/相位、机身位置/局部线角速度、双脚最低碰撞面高度、upright/跌倒标记、29关节q/dq/action与最后物理子步torque，不等同isaac_diag.csv字段。exp2.9起脚本还记录参考q/dq、PD目标、相位推进率、roll/pitch/yaw、足部接触标记和世界坐标接触力（接触坐标转换后按足部聚合，最后物理子步）；旧视频CSV没有这些新增列。脚本中的settled_mean_vx只是5.5秒以后跌倒前的样本均值，存在旋转/漂移时不能称稳态直行速度。

相位时钟依据checkpoint元数据选择：`phase_clock=command_scaled_v1`的exp2.9使用共享exp29_phase，起步按应用指令/native速度比例推进，参考dq按链式法则计算；旧模型保持原完整步频，不能用新时钟回放旧模型后直接与历史结果比较。原生全速周期保持1.26秒。

比较时使用同条件旧实验，区分训练均长、实际运动时长和回放跌倒时间；更晚跌倒不代表成功走路，单次续训比较不能隔离奖励改动的因果贡献。按用户指定写入 `czy/exp1/exp2.md`，不自动启动下一轮实验。
