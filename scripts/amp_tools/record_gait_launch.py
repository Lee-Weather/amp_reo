"""Record the authorized exp1.15 pair without duplicating experiment chapters."""
import json
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT/'czy/exp1/records'


def refresh():
    notebook = ROOT/'czy/exp1/exp1.md'
    text = notebook.read_text(encoding='utf-8')
    for exp in ('exp1.15', 'exp1.15t'):
        state = json.loads((RECORDS/(exp+'_training.json')).read_text())
        directory = RECORDS/(exp+'_evaluation')
        publication = json.loads((directory/'publication.json').read_text())
        verified = state.get('first_checkpoint_upload_verification', {}).get('verified', False)
        row = re.search(r'^\| '+re.escape(exp)+r' \|.*$', text, re.M)
        assert row
        parts = row[0].split('|')
        parts[4] = ' '+('待训练' if state['phase'] == 'created' else '训练中')+' '
        parts[5] = ' '+state['task_id']+' '
        if verified: parts[7] = ' model_100.pt（首模，最终模型待生成） '
        text = text[:row.start()]+'|'.join(parts)+text[row.end():]
        start = text.index('## 实验 '+exp+'：')
        end = text.find('\n## 实验 ', start+1)
        if end < 0: end = len(text)
        body = text[start:end]
        body = re.sub(r'^\| Git \|.*$', '| Git | '+publication['branch']+'，'+publication['commit']+' |', body, flags=re.M)
        body = re.sub(r'^\| Task ID / task \|.*$', '| Task ID / task | '+state['task_id']+' /'+state['task']+' |', body, flags=re.M)
        body = re.sub(r'启动命令（私有下载参数省略）：`[^`]+`', '启动命令（私有下载参数省略）：`'+state['start_command']+'`', body)
        marker = '\n#### 启动与首模验收\n'
        body = body.split(marker)[0].rstrip()
        body += marker+'\n实际任务 '+state['task_id']+'，账号 '+state['account']+'，代码 '+state['commit']+'；状态记录 '+state['phase']+'。\n'
        if state.get('replaces_failed_task'):
            body += '\n旧任务 '+state['replaces_failed_task']+' 的官方checkpoint挂载尝试在159秒后以文件缺失退出，实际0次训练更新；平台仅允许编辑草稿，保留该失败记录并创建当前替代任务。用户明确批准私有任务参数中使用教师签名下载地址；运行时在内存读取并核对SHA，不公开模型、不在本地记录地址；新任务预算仍为1000次更新。\n'
        if verified:
            evidence = state['first_checkpoint_upload_verification']
            receipt = evidence['save_receipt']
            body += f"\n首100已保存、SDK成功上传、平台精确条目{evidence['platform_model']['id']}确认，{receipt['bytes']}字节，SHA {receipt['sha256']}。验证时{evidence.get('latest_iteration')}轮，时间{evidence['checked_at']}。\n"
        inspection = directory/'first_checkpoint_inspection.json'
        if inspection.exists():
            item = json.loads(inspection.read_text())
            body += '\n首模已在本地CPU下载检查：哈希/真实4096、seed5、24步/2×4 PPO、D4、原控制与参考资源一致；参数和优化器全部有限，D计数2457600、完整DH JIT通过。教师组同时核对实际教师SHA及11250来源，未将输入当作输出。\n'
        progress = directory/'runtime_progress.json'
        if progress.exists():
            item = json.loads(progress.read_text())
            body += f"\n最新启动快照：{item['iteration']}/1000，采样{item['total_timesteps']}；AMP课程乘数{item.get('amp_curriculum_weight')}，mean episode length={item.get('mean_episode_length_steps')}步。该长度为已结束回合统计，含站立缓冲，不是连续行走时间。\n"
        body += '\n首阶段最终步态仍需1000轮完成后回放验收。\n'
        text = text[:start]+body+text[end:]
        plan_path = directory/'plan.json'
        plan = json.loads(plan_path.read_text())
        plan.update(status='training' if state['phase'] != 'created' else 'prepared_not_started',
                    task_id=state['task_id'], commit=state['commit'], first_checkpoint_verified=verified)
        plan_path.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    notebook.write_text(text, encoding='utf-8')
    print('Actual launch evidence refreshed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    if parser.parse_args().refresh:
        return refresh()
    notebook = ROOT/'czy/exp1/exp1.md'
    text = notebook.read_text(encoding='utf-8')
    if '## 实验 exp1.15：' in text:
        raise ValueError('Chapters already exist; update their actual status instead of duplicating')
    rows = []
    for route in ('115', '115t'):
        exp = 'exp1.15'+('t' if route.endswith('t') else '')
        state = json.loads((RECORDS/(exp+'_training.json')).read_text())
        status = '训练中' if state['phase'] != 'created' else '待训练'
        rows.append(f"| {exp} | 2026-10-30 | 相同正向课程与归一化/时间窗引导；{'精确迁移1.9动作函数的DH对照' if route.endswith('t') else '继续学习14的稳定DH策略'}，首阶段1000轮 | {status} | {state['task_id']} | {state['account']} | 待生成 |")
    text = text.replace('\n当前命名规则', '\n'+'\n'.join(rows)+'\n\n当前命名规则', 1)
    text = text.replace('当前实施exp1.14及exp1.14t两条独立算法路线，均从同一exp1.13t最终策略迁移；本轮t不表示从零对照。',
        '当前实施exp1.15及exp1.15t：共同修正奖励并采用正向课程，比较稳定DH继续学习与精确教师动作函数迁移；本轮t不表示从零对照。14/14t额度提前终止后已完成回放。', 1)
    text = text.replace('该方案尚未实施，不在回放流程中自动推送或开训。',
        '该段为回放结束时的方案；2026-10-30用户随后明确授权实施并指定新账号，实施见下方exp1.15/exp1.15t。', 1)
    for route in ('115', '115t'):
        exp = 'exp1.15'+('t' if route.endswith('t') else '')
        teacher = route.endswith('t')
        directory = RECORDS/(exp+'_evaluation')
        state = json.loads((RECORDS/(exp+'_training.json')).read_text())
        publication = json.loads((directory/'publication.json').read_text())
        preflight = json.loads((directory/'cpu_preflight.json').read_text())
        initialization = ('先继承同一exp1.14/model_5300.pt的CNN、速度估计器和log_std，再将exp1.9/model_11250.pt的Actor精确映射到DH的五帧核心短历史。5×96映射到5×98，相位/估计速度/CNN的Actor输入列初始置零，其他Actor层直接复制；后续全部分支仍可训练。保留稳定DH的探索噪声，不能声称同时复制了旧教师的随机采样分布。'
                          if teacher else '保留exp1.14真实model_5300.pt的Actor、CNN、速度估计器和log_std。Critic、D/normalizer、两个Adam、回放缓存均重新建立；本轮从0计1000次新更新。')
        text += f'''

## 实验 {exp}：{'前进教师初始化的DH对照' if teacher else '归一化时间窗引导的DH正向课程'}

### 1. 上一实验结果与教训

exp1.14真实5300主回放0次跌倒，但0.6指令实际vx0.000254m/s、RMSE0.599855m/s；14t真实5600同样不前进，左膝原始目标裁剪95.23%。两组双脚低于1cm均100%，RSI60秒各11次跌倒。固定姿态在旧0.3指令参考奖励中仍约有80%信用，存活或参考频率微动不能替代迈步。exp1.9的11250在0.65指令下vx0.6321、RMSE0.0436m/s，能作为前进/零速能力来源，但后退和整体视觉验收未通过。

### 2. 本轮修改目标

先解决站姿正向起步，共用奖励/指令分布/预算，比较初始化。本阶段每组1000轮，之后先回放验收；+0.3m/s连续60秒，mean vx0.25–0.35、RMSE≤0.10、无非超时跌倒，并检查交替迈步、脚底离地和自然姿态。通过后再讨论扩大正向速度，最后才加后退；不自动续更多轮或进入exp2。

### 3. 修改内容

### 修改一：参考误差按实际动作幅度归一化

q误差除以参考周期标准差，dq除以参考RMS，足端位置除以参考标准差，公差为0.5倍量级，并设小幅维度下限。旧0.35rad/1.5rad/s/0.06m固定宽容尺度不再控制本轮评分。q/足端正奖励乘实际时间窗活动量；dq评分减去同一参考相位dq=0的静止分数，保留负值。相位输入与真实参考周期继续沿用exp1.14。

### 修改二：增加时间窗幅度、抬脚支撑、净位移

记录2秒（101个50Hz样本）的六个腿pitch、两脚相对机身x、世界位置和朝向，按参考峰峰值比较实际幅度并扣除静止基线；过大幅度可为负。摆动侧需有实际脚底离地，对侧需20N支撑。净位移按窗口开始朝向投影，采用归一化Huber进度奖励；原地摇摆净位移为0。每次重置或实际指令改变都清空窗口计数，窗口完整前不产生幅度/净位移信用。足端使用当前rigid_state，避免尚未刷新AMP历史造成一帧延迟。

| 奖励 | 权重 | 约束 |
| --- | --- | --- |
| reference_pose | 2 | 归一化、实际活动量加权 |
| reference_velocity | 1 | 归一化、扣除静止dq基线 |
| reference_feet | 1 | 归一化、实际活动量加权 |
| reference_amplitude | 2 | 时间窗幅度优于静止基线 |
| reference_support | 1 | 参考摆动侧离地、另一脚支撑 |
| net_displacement | 2 | 两秒净位移匹配指令 |

保留原tracking、motion_progress和站立/控制/能耗/姿态约束，stand_pose=-1.2、raw_target_violation=-2。引导总权重在前500轮从0.25渐升至1。新增引导仍以站稳/高度/非跌倒状态为有效条件，不把实际速度设成开始学习的硬门槛。

### 修改三：固定正向课程，暂缓AMP对策略的奖励

请求事件80%为+0.3m/s、20%为0，vy=wz=0。仍有1.5秒站立缓冲，所以实际应用比例需查日志。前500轮AMP策略奖励乘数为0，500→1500线性恢复为1，首阶段1000轮结束时为0.5；D仍训练。原四段AMP专家、RSI=0.5保留。显式回放指令独立于训练采样，并在首次RSI重置前生效。

### 修改四：明确初始化

{initialization}

两组均为66×98=6468观测、5帧短历史490、Actor输入557、Critic101、29动作，原Kp/Kd、50Hz、动作比例0.25、无LPF。t只改变Actor初始化，不作为从零训练对照。

### 4. 修改文件

- humanoid/amp/gait_guidance.py：参考量级、时间窗和静止基线。
- humanoid/envs/x1/x1_amp_exp115_config.py、x1_amp_exp115_env.py、x1_amp_exp115t_config.py：共同奖励/课程和两种初始化。
- humanoid/amp/teacher_transfer.py、humanoid/scripts/train_guided_amp.py：教师约定校验、精确迁移与私有输入；checkpoint.py校验共享目录的额外输入哈希。
- runner.py：教师缺失时拒绝静默开训，原runner日志统计保持；任务注册、play/smoke和CPU预检/测试同步。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 账号 / userId / 项目 | {state['account']} /4684 /{state['project_id']} |
| Task ID / task | {state['task_id']} /{state['task']} |
| 新轮数 / 保存间隔 | 1000 /100，计数从0 |
| num_envs / seed | 4096 /5 |
| rollout / PPO / D | 24 /2epochs×4小批 /4次 |
| 新采样预算 | 98,304,000 |
| lr / 探索 | 前200轮上限1e-5，随后自适应≤1e-4；entropy0.001，std上限0.5，四髋膝moving std下限0.25 |
| 算力 / 镜像 | ESKU000001，1×4090D24G，5.76元/小时 /BJX00000001 V000412 |
| Git | {publication['branch']}，{publication['commit']} |
| 源模型 | exp1.14/model_5300.pt，SHA {state['source_checkpoint_sha256']} |

启动命令（私有下载参数省略）：`{state['start_command']}`。本轮输出model_100.pt至model_1000.pt，不能把输入5300/11250当作训练输出。t教师先上传官方个人存储且下载校验，但4090D不支持挂载该存储，平台在运行前拒绝；保留同一已创建任务，改走官方checkpoint私有输入通道。私有ACL显式设置并验证匿名读403，模型不进入Git或公开Release。账号密钥和签名URL不写记录。

### 6. 预期与验收

本地仅CPU：准确提交回归、真实生产模型、4096条分批推理、真实PPO/AMP参数更新、保存恢复、完整JIT与静止反例。t额外执行真实云入口的双输入同目录启动流程，Gym注册表由CPU fixture代替。开训验收要求实际账号4684/commit一致、真实迭代、首100模型receipt/SDK上传/平台条目、下载哈希和模型内容检查。正式1000轮后验收本阶段0.3起步、零速和RSI；不把未训练后退当作阶段起步指标，也不宣称全范围完成。

### 7. 实验结果

准确提交CPU回归及生产模型预检通过。初始动作对照最大误差{preflight['inference_max_error']:.9g}，完整JIT误差{preflight['complete_jit_max_error']:.9g}。新增引导的运动学反例：静止和周期平均固定姿态均0，微抖约0.00619，完整原地步态约6.69789，参考步态加净前进约8.69789；这些只表示新增引导，不是总回报或真实物理验收。实际训练进度/首模证据见对应evaluation和training.json，最终步态结果待1000轮结束后补充。
'''
    notebook.write_text(text, encoding='utf-8')
    print('Experiment index and both seven-section chapters recorded')


if __name__ == '__main__': main()
