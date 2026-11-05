"""Start exp2.3 once, with a private verified model950 input."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import flux_ops as flux
from private_training_input import request

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 'limx1c186cc3vzmj@maxxspace.com'
SOURCE_ACCOUNT = 'limx1c186cc3vzmj@maxxspace.com'
SOURCE_TASK = 'TASK_20261102_130'
SOURCE_SHA = 'b284d5558826b781de1b1baa263b7e472207cf4e59dd94b74a97f49dabe89b51'
BRANCH = 'codex/exp2-3'
PROXY = 'socks5h://10.12.201.122:39009'


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def create(kind, payload):
    # The CLI requires a file. Its sensitive contents are short-lived and the
    # full dry-run/result is never printed or written to permanent records.
    with tempfile.TemporaryDirectory(prefix='.flux-exp23-',dir=ROOT) as temporary:
        path = Path(temporary)/'request.json'
        path.write_text(json.dumps(payload),encoding='utf-8')
        relative = path.relative_to(ROOT).as_posix()
        assert flux.call('--dry-run',kind,'create','--file',relative)['success']
        data = flux.call(kind,'create','--file',relative)['data']
    identifier = data if isinstance(data,str) else data.get(kind+'Id') or data.get('id')
    assert identifier, 'Creation returned; inspect platform before any retry'
    return identifier


def main():
    directory = ROOT/'czy/exp1/records/exp2.3_evaluation'
    state_path = ROOT/'czy/exp1/records/exp2.3_training.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        assert previous['task_id']=='TASK_20261102_134' and (directory/'startup_failure.json').exists()
        flux.ACCOUNT=ACCOUNT
        info=flux.call('task','info','--task-id',previous['task_id'])['data']['taskBaseInfo']
        assert str(info['taskStatus'])=='6' and int(info['userId'])==4668

    publication = json.loads((directory/'continuation_fix_publication.json').read_text())
    assert publication['published'] and publication['exact_isolated_tree_cpu_preflight']
    commit = publication['remote_commit']
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    remote = subprocess.check_output(['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl',
        'ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    assert hashlib.sha256((ROOT/'czy/data/exp2.2/model_950.pt').read_bytes()).hexdigest()==SOURCE_SHA
    flux.ACCOUNT = SOURCE_ACCOUNT
    who = flux.call('auth','whoami')['data']
    assert who['email']==SOURCE_ACCOUNT and int(who['userId'])==4668
    rows = flux.call('task','model','list','--task-id',SOURCE_TASK,'--checkpoint','950','--limit','30')['data']['rows']
    row = next(r for r in rows if r['fileName']=='model_950.pt' and str(r['checkpoint'])=='950')
    status, _, raw = request(row['policUrlDown'])
    assert status==200 and hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    encoded = base64.urlsafe_b64encode(row['policUrlDown'].encode()).decode()
    write(directory/'source_input_verification.json',dict(source_task_id=SOURCE_TASK,
        source_account=SOURCE_ACCOUNT,file_name='model_950.pt',bytes=len(raw),sha256=SOURCE_SHA,
        downloaded_and_hash_verified=True,signed_urls_persisted=False))
    del raw
    flux.ACCOUNT = ACCOUNT
    who = flux.call('auth','whoami')['data']
    assert who['email']==ACCOUNT and int(who['userId'])==4668
    tasks=flux.call('task','list','--task-name','exp2.3-','--limit','50')['data']['rows']
    assert all(r['taskId']=='TASK_20261102_134' and str(r['taskStatus'])=='6' for r in tasks), 'Inspect other existing task'
    resources = flux.call('task','resource','list','--goods-back-category','3','--limit','20')['data']['rows']
    goods = next(r for r in resources if r['goodsId']=='ESKU000001')
    assert goods['goodsBackSku']['gpuNum']==1 and goods['goodsBackSku']['gpuModel']=='4090D'
    assert any(r['imageId']=='BJX00000001' for r in flux.call('task','image','official')['data']['rows'])
    assert any(r['id']=='V000412' for r in flux.call('task','image','versions','--image-id','BJX00000001')['data'])
    project_name = 'X1 29DOF AMP exp2.3 relaxed anchor'
    projects = flux.call('project','list','--limit','100')['data']['rows']
    matches = [r for r in projects if r['projectName']==project_name]
    assert len(matches)<=1
    project_id = matches[0]['projectId'] if matches else create('project',dict(projectName=project_name,
        projectDescription='exp2.3: native0.595m/s full-body AMP, policy950 transfer and200 new updates.'))
    project = flux.call('project','info','--project-id',project_id)['data']
    assert int(project['userId'])==4668 and project['projectName']==project_name
    write(directory/'project.json',dict(project_id=project_id,project_name=project_name,account=ACCOUNT,user_id=4668))
    checkpoint_dir = '/workspace/amp_reo/logs/x1_amp_exp23/exported_data/exp2.3_main'
    command = ('gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp_exp23 --headless --num_envs=4096'
               ' --seed=5 --max_iterations=200 --run_name=exp2.3 --checkpoint_dir='+checkpoint_dir+
               ' --warm_start='+checkpoint_dir+'/input_model_950.pt --warm_start_sha256='+SOURCE_SHA)
    state = dict(experiment='exp2.3',account=ACCOUNT,account_identity_verified=dict(email=ACCOUNT,user_id=4668),
        task='x1_amp_exp23',branch=BRANCH,commit=commit,project_id=project_id,
        source_task_id=SOURCE_TASK,source_checkpoint='czy/data/exp2.2/model_950.pt',source_iteration=950,
        source_checkpoint_sha256=SOURCE_SHA,training_mode='anchored_continuation_preserve_critic_optimizers',
        new_initial_iteration=0,max_iterations=200,additional_updates=200,target_iteration=200,
        num_envs=4096,seed=5,num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=4,
        discriminator_updates_per_iteration=1,sample_budget=200*4096*24,
        source_anchor_coef=1.,source_max_pd_rms=.04,exploration_gain=.5,critic_only_iterations=0,
        amp_paused_first_new_updates=200,continuation_budget_not_submitted=800,
        observation_dim=6468,action_dim=29,checkpoint_dir=checkpoint_dir,start_command=command,
        first_checkpoint_iteration=50,goods_id='ESKU000001',image_id='BJX00000001',image_version='V000412',
        current_goods_price_per_hour=goods['goodsPrice'],command_native_speed=.5953696966,
        phase='validated_not_created',signed_urls_persisted_locally=False)
    write(directory/'plan.json',state)
    payload = dict(taskBaseInfo=dict(projectId=project_id,taskType='1',trainType='1',
        taskName='exp2.3-relaxed-anchor-4096-seed5-check200',
        taskDescription='Continue exp2.2 model950: preserve critic,Adam,original7000 anchor;reset new counter andcurriculum. '
            'Source soft coef1;PD RMS guard0.04rad;exploration0.5;CNN/estimator/std frozen. '
            'AMP paused first200;then two qualifying windows allow0.01. '
            'Only200 NEW updates authorized;4096env seed5 24rollout2x4PPO save50.',
        taskTag=['exp2.3','AMP','native-gait'],goodsId='ESKU000001',imageId='BJX00000001',
        imageVersion='V000412',personalDataPath=''),
        taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/amp_reo.git',
            versionType='1',versionName=BRANCH)]),mainCodeUri='amp_reo/humanoid/scripts/train.py',
            hparamsPath='amp_reo/humanoid/envs/x1/x1_amp_exp23_config.py',
            startScript=command+' --warm_start_url_b64='+encoded,isOpen='0'),
        runtimeReminderConfig=dict(enableRuntimeReminder=False,reminderDurations=[]))
    task_id = create('task',payload)
    state.update(task_id=task_id,phase='created',created_at=datetime.now(timezone.utc).isoformat())
    write(state_path,state)
    print('TASK_CREATED '+task_id,flush=True)
    assert flux.call('--dry-run','task','run','--task-id',task_id)['success']
    flux.call('task','run','--task-id',task_id)
    state.update(phase='submitted',submitted_at=datetime.now(timezone.utc).isoformat())
    write(state_path,state); write(directory/'plan.json',state)
    print(json.dumps({k:state[k] for k in ('experiment','task_id','phase','account','project_id','commit')}),flush=True)


if __name__=='__main__': main()
