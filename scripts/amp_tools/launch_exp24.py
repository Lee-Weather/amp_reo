"""Start exp2.4 once, with a private verified model950 input."""
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
SOURCE_TASK = 'TASK_20261102_135'
SOURCE_SHA = 'e453506b82725d470745993bb703475099be4411b02eb57df182ab9eaf3bf2fa'
BRANCH = 'codex/exp2-4'
PROXY = 'socks5h://10.12.201.122:39009'


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def create(kind, payload):
    # The CLI requires a file. Its sensitive contents are short-lived and the
    # full dry-run/result is never printed or written to permanent records.
    with tempfile.TemporaryDirectory(prefix='.flux-exp24-',dir=ROOT) as temporary:
        path = Path(temporary)/'request.json'
        path.write_text(json.dumps(payload),encoding='utf-8')
        relative = path.relative_to(ROOT).as_posix()
        assert flux.call('--dry-run',kind,'create','--file',relative)['success']
        data = flux.call(kind,'create','--file',relative)['data']
    identifier = data if isinstance(data,str) else data.get(kind+'Id') or data.get('id')
    assert identifier, 'Creation returned; inspect platform before any retry'
    return identifier


def main():
    directory = ROOT/'czy/exp1/records/exp2.4_evaluation'
    state_path = ROOT/'czy/exp1/records/exp2.4_training.json'
    assert not state_path.exists(), 'Inspect existing task; never create a duplicate'

    publication = json.loads((directory/'publication.json').read_text())
    assert publication['published'] and publication['exact_isolated_tree_cpu_preflight']
    commit = publication['remote_commit']
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    remote = subprocess.check_output(['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl',
        'ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    assert hashlib.sha256((ROOT/'czy/data/exp2.3/model_200.pt').read_bytes()).hexdigest()==SOURCE_SHA
    flux.ACCOUNT = SOURCE_ACCOUNT
    who = flux.call('auth','whoami')['data']
    assert who['email']==SOURCE_ACCOUNT and int(who['userId'])==4668
    rows = flux.call('task','model','list','--task-id',SOURCE_TASK,'--checkpoint','200','--limit','30')['data']['rows']
    row = next(r for r in rows if r['fileName']=='model_200.pt' and str(r['checkpoint'])=='200')
    status, _, raw = request(row['policUrlDown'])
    assert status==200 and hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    encoded = base64.urlsafe_b64encode(row['policUrlDown'].encode()).decode()
    write(directory/'source_input_verification.json',dict(source_task_id=SOURCE_TASK,
        source_account=SOURCE_ACCOUNT,file_name='model_200.pt',bytes=len(raw),sha256=SOURCE_SHA,
        downloaded_and_hash_verified=True,signed_urls_persisted=False))
    del raw
    flux.ACCOUNT = ACCOUNT
    who = flux.call('auth','whoami')['data']
    assert who['email']==ACCOUNT and int(who['userId'])==4668
    tasks=flux.call('task','list','--task-name','exp2.4-','--limit','50')['data']['rows']
    assert not tasks, 'Inspect existing task before creating any duplicate'
    resources = flux.call('task','resource','list','--goods-back-category','3','--limit','20')['data']['rows']
    goods = next(r for r in resources if r['goodsId']=='ESKU000001')
    assert goods['goodsBackSku']['gpuNum']==1 and goods['goodsBackSku']['gpuModel']=='4090D'
    assert any(r['imageId']=='BJX00000001' for r in flux.call('task','image','official')['data']['rows'])
    assert any(r['id']=='V000412' for r in flux.call('task','image','versions','--image-id','BJX00000001')['data'])
    project_name = 'X1 29DOF AMP exp2.4 overnight AMP'
    projects = flux.call('project','list','--limit','100')['data']['rows']
    matches = [r for r in projects if r['projectName']==project_name]
    assert len(matches)<=1
    project_id = matches[0]['projectId'] if matches else create('project',dict(projectName=project_name,
        projectDescription='exp2.4: native0.595m/s full-body AMP, policy200 continuation and4000 additional updates.'))
    project = flux.call('project','info','--project-id',project_id)['data']
    assert int(project['userId'])==4668 and project['projectName']==project_name
    write(directory/'project.json',dict(project_id=project_id,project_name=project_name,account=ACCOUNT,user_id=4668))
    checkpoint_dir = '/workspace/amp_reo/logs/x1_amp_exp24/exported_data/exp2.4_main'
    command = ('gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp_exp24 --headless --num_envs=4096'
               ' --seed=5 --max_iterations=4000 --run_name=exp2.4 --checkpoint_dir='+checkpoint_dir+
               ' --warm_start='+checkpoint_dir+'/input_model_200.pt --warm_start_sha256='+SOURCE_SHA)
    state = dict(experiment='exp2.4',account=ACCOUNT,account_identity_verified=dict(email=ACCOUNT,user_id=4668),
        task='x1_amp_exp24',branch=BRANCH,commit=commit,project_id=project_id,
        source_task_id=SOURCE_TASK,source_checkpoint='czy/data/exp2.3/model_200.pt',source_iteration=200,
        source_checkpoint_sha256=SOURCE_SHA,training_mode='anchored_continuation_preserve_critic_optimizers',
        new_initial_iteration=200,max_iterations=4000,additional_updates=4000,target_iteration=4200,
        num_envs=4096,seed=5,num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=4,
        discriminator_updates_per_iteration=1,sample_budget=4000*4096*24,
        source_anchor_coef=1.,source_max_pd_rms=.04,exploration_gain=.5,critic_only_iterations=0,
        discriminator_preheat_updates=100,amp_stage_weights=[0.,.01,.03,.05],max_wall_seconds=19800,
        observation_dim=6468,action_dim=29,checkpoint_dir=checkpoint_dir,start_command=command,
        first_checkpoint_iteration=250,goods_id='ESKU000001',image_id='BJX00000001',image_version='V000412',
        current_goods_price_per_hour=goods['goodsPrice'],command_native_speed=.5953696966,
        phase='validated_not_created',signed_urls_persisted_locally=False)
    write(directory/'plan.json',state)
    payload = dict(taskBaseInfo=dict(projectId=project_id,taskType='1',trainType='1',
        taskName='exp2.4-relaxed-anchor-4096-seed5-plus4000',
        taskDescription='Continue exp2.3 model200 to4200: preserve critic,Adam,original7000 anchor andcounter. '
            'D trains fromstart;AMP0 until100D updates andtwo stable windows,then0.01/0.03/0.05. '
            'Softcoef1;PD RMSguard0.04rad;exploration0.5;CNN/estimator/std frozen. '
            '4096env seed5 24rollout2x4PPO save100+first50+best;maxruntime5.5h;stagnation/regression stops.',
        taskTag=['exp2.4','AMP','native-gait'],goodsId='ESKU000001',imageId='BJX00000001',
        imageVersion='V000412',personalDataPath=''),
        taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/amp_reo.git',
            versionType='1',versionName=BRANCH)]),mainCodeUri='amp_reo/humanoid/scripts/train.py',
            hparamsPath='amp_reo/humanoid/envs/x1/x1_amp_exp24_config.py',
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
