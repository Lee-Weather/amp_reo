"""Launch one approved private exp2.10 task on the explicitly selected account."""
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import flux_ops as flux
from launch_exp29 import create,write,PROXY
from private_training_input import request

ROOT=Path(__file__).resolve().parents[2]
ACCOUNT='limx1c2a7jqk61ku@maxxspace.com'
SOURCE_ACCOUNT='limx1c210k7rhm75@maxxspace.com'
SOURCE_TASK='TASK_20261103_210'
SHA='23948ebc7492c96754b8eece068be80418356206588e1e7950bc13da8decb9b0'
NAME='exp2.10-yaw-only-4096-seed5-plus1000'
BRANCH='codex/exp2-10'
DIRECTORY=ROOT/'czy/exp1/records/exp2.10_evaluation'
STATE=ROOT/'czy/exp1/records/exp2.10_training.json'
EXPERIMENT='exp2.10'
TASK='x1_amp_exp210'
ENTRY='train_yaw_amp.py'
TRANSFER_MODULE='exp210_transfer.py'
REFINEMENT='exp210_yaw_only_v1'
PROJECT_NAME='X1 29DOF AMP exp2.10 yaw stability'
DESCRIPTION='Full13500 to14500 +1000updates, only corridor_yaw penalty -0.10 to-0.30; preserve Actor/Critic,D13300,normalizer,bothAdam,original7000anchor,phase,AMP0.01,PD and all other rewards/physics. 4096 seed5 24rollout2x4;save100,first50,best;original regression and5.5h guards.'
CHANGE_FIELDS=dict(reward_change=dict(corridor_yaw=[-.10,-.30]))


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    flux.ACCOUNT=ACCOUNT
    who=flux.call('auth','whoami')['data']
    assert who['email']==ACCOUNT and int(who['userId'])==4670
    if STATE.exists():
        state=json.loads(STATE.read_text())
        info=flux.call('task','info','--task-id',state['task_id'])['data']
        assert int(info['taskBaseInfo']['userId'])==4670
        if str(info['taskBaseInfo']['taskStatus'])=='0':
            assert flux.call('--dry-run','task','run','--task-id',state['task_id'])['success']
            flux.call('task','run','--task-id',state['task_id'])
            state.update(phase='submitted',submitted_at=datetime.now(timezone.utc).isoformat());write(STATE,state)
        print(json.dumps(dict(task_id=state['task_id'],existing_task=True)),flush=True)
        return
    assert not flux.call('task','list','--task-name',NAME,'--limit','50')['data']['rows'],'Inspect existing task; do not duplicate'
    publication=json.loads((DIRECTORY/'publication.json').read_text())
    assert publication['published'] and publication['exact_isolated_tree_cpu_preflight']
    commit=publication['remote_commit']
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    remote=subprocess.check_output(['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl',
        'ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    for file in ('humanoid/scripts/'+ENTRY,'humanoid/scripts/resume_amp.py',
                 'humanoid/amp/'+TRANSFER_MODULE,'humanoid/envs/x1/'+TASK+'_config.py'):
        subprocess.run(['git','cat-file','-e',commit+':'+file],check=True,capture_output=True)
    flux.ACCOUNT=SOURCE_ACCOUNT
    source_who=flux.call('auth','whoami')['data'];assert source_who['email']==SOURCE_ACCOUNT and int(source_who['userId'])==4669
    source_info=flux.call('task','info','--task-id',SOURCE_TASK)['data']
    assert str(source_info['taskBaseInfo']['taskStatus'])=='5' and int(source_info['taskBaseInfo']['userId'])==4669
    rows=flux.call('task','model','list','--task-id',SOURCE_TASK,'--checkpoint','13500','--limit','30')['data']['rows']
    row=next(r for r in rows if r['fileName']=='model_13500.pt' and str(r['checkpoint'])=='13500')
    http,headers,raw=request(row['policUrlDown']);assert http==200
    assert hashlib.sha256(raw).hexdigest()==SHA and len(raw)==int(headers['content-length'])
    assert hashlib.md5(raw).hexdigest()==headers['etag'].strip('"').lower()
    model=ROOT/'czy/data/exp2.9/model_13500.pt';assert hashlib.sha256(model.read_bytes()).hexdigest()==SHA
    encoded=base64.urlsafe_b64encode(row['policUrlDown'].encode()).decode()
    write(DIRECTORY/'source_input_verification.json',dict(source_task=SOURCE_TASK,source_account=SOURCE_ACCOUNT,
        target_account=ACCOUNT,source_iteration=13500,sha256=SHA,bytes=len(raw),downloaded_and_hash_verified=True,
        etag_verified=True,private_url_not_in_permanent_records=True))
    del raw
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(model,map_location='cpu',weights_only=False)
    assert checkpoint['iter']==13500 and checkpoint['infos']['exp21']['learning_state']['curriculum']['d_updates']==13300
    flux.ACCOUNT=ACCOUNT
    resources=flux.call('task','resource','list','--goods-back-category','3','--limit','20')['data']['rows']
    goods=next(r for r in resources if r['goodsId']=='ESKU000001')
    assert goods['goodsBackSku']['gpuNum']==1 and goods['goodsBackSku']['gpuModel']=='4090D'
    assert any(r['imageId']=='BJX00000001' for r in flux.call('task','image','official')['data']['rows'])
    assert any(r['id']=='V000412' for r in flux.call('task','image','versions','--image-id','BJX00000001')['data'])
    project_name=PROJECT_NAME
    projects=flux.call('project','list','--limit','100')['data']['rows']
    matches=[p for p in projects if p['projectName']==project_name]
    assert len(matches)<=1
    project_id=matches[0]['projectId'] if matches else create('project',dict(projectName=project_name,
        projectDescription=DESCRIPTION))
    project=flux.call('project','info','--project-id',project_id)['data'];assert int(project['userId'])==4670
    directory='/workspace/amp_reo/logs/'+TASK+'/exported_data/'+EXPERIMENT+'_main'
    command=('gm-run amp_reo/humanoid/scripts/'+ENTRY+' --task='+TASK+' --headless --num_envs=4096'
        ' --seed=5 --max_iterations=1000 --run_name='+EXPERIMENT+' --recovery_model='+directory+
        '/input_model_13500.pt --recovery_sha256='+SHA+' --source_iteration=13500 --target_iteration=14500')
    state=dict(experiment=EXPERIMENT,account=ACCOUNT,account_identity_verified=dict(email=ACCOUNT,user_id=4670),
        source_account=SOURCE_ACCOUNT,source_task_id=SOURCE_TASK,project_id=project_id,task=TASK,
        commit=commit,branch=BRANCH,parent_algorithm_commit=publication['parent'],
        training_mode='anchored_continuation_preserve_critic_optimizers',resume_mode='full_checkpoint',refinement=REFINEMENT,
        source_checkpoint='czy/data/exp2.9/model_13500.pt',source_checkpoint_sha256=SHA,
        policy_origin_source_sha256=checkpoint['training_origin']['source_sha256'],
        source_iteration=13500,new_initial_iteration=13500,target_iteration=14500,
        max_iterations=1000,additional_updates=1000,recovery_additional_updates=1000,
        num_envs=4096,seed=5,observation_dim=6468,action_dim=29,checkpoint_dir=directory,start_command=command,
        first_checkpoint_iteration=13550,source_discriminator_updates=13300,max_wall_seconds=19800,
        num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=4,sample_budget=98304000,
        amp_weight=.01,current_goods_price_per_hour=goods['goodsPrice'],
        phase='prepared',created_at=datetime.now(timezone.utc).isoformat(),signed_urls_in_permanent_records=False)
    state.update(CHANGE_FIELDS)
    write(DIRECTORY/'plan.json',state)
    payload=dict(taskBaseInfo=dict(projectId=project_id,taskType='1',trainType='1',taskName=NAME,
        taskDescription=DESCRIPTION,
        taskTag=[EXPERIMENT,'AMP','sim2sim'],goodsId='ESKU000001',imageId='BJX00000001',imageVersion='V000412',personalDataPath=''),
        taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/amp_reo.git',versionType='1',versionName=BRANCH)]),
        mainCodeUri='amp_reo/humanoid/scripts/'+ENTRY,hparamsPath='amp_reo/humanoid/envs/x1/'+TASK+'_config.py',
        startScript=command+' --recovery_url_b64='+encoded,isOpen='0'),
        runtimeReminderConfig=dict(enableRuntimeReminder=False,reminderDurations=[]))
    task_id=create('task',payload)
    state.update(task_id=task_id,phase='created');write(STATE,state)
    print('TASK_CREATED '+task_id,flush=True)
    assert flux.call('--dry-run','task','run','--task-id',task_id)['success']
    flux.call('task','run','--task-id',task_id)
    state.update(phase='submitted',submitted_at=datetime.now(timezone.utc).isoformat());write(STATE,state)
    print(json.dumps({k:state[k] for k in ('task_id','account','phase','commit','source_iteration','target_iteration')}),flush=True)


if __name__=='__main__':main()
