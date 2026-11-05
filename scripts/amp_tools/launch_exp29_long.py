"""Start one approved5000-update continuation only after exp2.9 completes."""
import base64
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import flux_ops as flux
from fetch_final_amp import final_model, validate_checkpoint
from launch_exp29 import create, write, PROXY, ACCOUNT
from private_training_input import request

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'czy/exp1/records/exp2.9_long_evaluation'
STATE=ROOT/'czy/exp1/records/exp2.9_long_training.json'
ORIGINAL=ROOT/'czy/exp1/records/exp2.9_training.json'
PARENT_COMMIT='66a989f7327b4fa560b86d7a418122de72416a30'
BRANCH='codex/exp2-9-long'
NAME='exp2.9-long-r1-4096-seed5-plus5000'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-failed-task')
    args=parser.parse_args()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    flux.ACCOUNT=ACCOUNT
    who=flux.call('auth','whoami')['data']
    assert who['email']==ACCOUNT and int(who['userId'])==4669
    parent=json.loads(ORIGINAL.read_text())
    assert parent['commit']==PARENT_COMMIT and parent['task_id']=='TASK_20261103_172'
    replacement=None
    if STATE.exists():
        state=json.loads(STATE.read_text())
        info=flux.call('task','info','--task-id',state['task_id'])['data']
        status=str(info['taskBaseInfo']['taskStatus'])
        if args.replace_failed_task==state['task_id']:
            assert state['task_id']=='TASK_20261103_192' and status=='6'
            assert not flux.call('task','model','list','--task-id',state['task_id'],'--limit','10')['data']['rows']
            replacement=state['task_id']
            archive=DIRECTORY/('failed_'+replacement+'_training.json')
            assert not archive.exists() or json.loads(archive.read_text())==state
            write(archive,state)
            proof=DIRECTORY/'first_checkpoint_verification.json'
            if proof.exists():
                archived=DIRECTORY/('failed_'+replacement+'_first_checkpoint.json')
                if not archived.exists():write(archived,json.loads(proof.read_text()))
        elif args.replace_failed_task:
            raise ValueError('Requested replacement does not match existing state; inspect actual task')
        elif status=='0':
            assert flux.call('--dry-run','task','run','--task-id',state['task_id'])['success']
            flux.call('task','run','--task-id',state['task_id'])
            state.update(phase='submitted',submitted_at=datetime.now(timezone.utc).isoformat())
            write(STATE,state)
        if replacement is None:
            print(json.dumps(dict(task_id=state['task_id'],phase=state['phase'],existing_task=True)),flush=True)
            return
    info=flux.call('task','info','--task-id',parent['task_id'])['data']
    assert str(info['taskBaseInfo']['userId'])=='4669' and info['taskCodeInfo']['commitId']==PARENT_COMMIT
    status=str(info['taskBaseInfo']['taskStatus'])
    if status!='5':
        if status=='6':raise RuntimeError('Original task terminated before normal completion; do not auto-resume')
        print('WAITING_ORIGINAL_COMPLETION',flush=True)
        return
    rows=flux.call('task','model','list','--task-id',parent['task_id'],'--checkpoint','8500','--limit','30')['data']['rows']
    try: row=final_model(rows,8500)
    except RuntimeError:
        print('WAITING_FINAL_MODEL_UPLOAD',flush=True)
        return
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    publication=json.loads((DIRECTORY/'publication_entry_fix.json').read_text())
    assert publication['published'] and publication['exact_isolated_tree_cpu_preflight']
    commit=publication['remote_commit']
    assert publication['parent']==PARENT_COMMIT and publication['branch']==BRANCH
    subprocess.run(['git','cat-file','-e',commit+':humanoid/scripts/resume_amp.py'],check=True,capture_output=True)
    remote=subprocess.check_output(['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl',
        'ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    http,headers,raw=request(row['policUrlDown']);assert http==200
    assert len(raw)==int(headers['content-length'])
    assert hashlib.md5(raw).hexdigest()==headers['etag'].strip('"').lower()
    digest=hashlib.sha256(raw).hexdigest()
    model=ROOT/'czy/data/exp2.9/model_8500.pt';model.parent.mkdir(parents=True,exist_ok=True)
    assert not model.exists() or hashlib.sha256(model.read_bytes()).hexdigest()==digest
    model.write_bytes(raw)
    import torch
    torch.set_num_threads(1)
    validation=validate_checkpoint(model,parent)
    checkpoint=torch.load(model,map_location='cpu',weights_only=False)
    assert checkpoint['metadata']['training']['phase_clock']=='command_scaled_v1'
    assert checkpoint['infos']['exp21']['learning_state']['curriculum']['d_updates']==8300
    write(DIRECTORY/'final_source_verification.json',dict(verified=True,source_iteration=8500,
        source_checkpoint=model.relative_to(ROOT).as_posix(),sha256=digest,bytes=len(raw),
        etag_md5_verified=True,original_completed=True,optimizer_steps=validation['optimizer_steps']))
    del raw
    assert not flux.call('task','list','--task-name',NAME,'--limit','50')['data']['rows'], 'Inspect existing task; do not create duplicate'
    resources=flux.call('task','resource','list','--goods-back-category','3','--limit','20')['data']['rows']
    goods=next(r for r in resources if r['goodsId']=='ESKU000001')
    assert goods['goodsBackSku']['gpuNum']==1 and goods['goodsBackSku']['gpuModel']=='4090D'
    assert any(r['imageId']=='BJX00000001' for r in flux.call('task','image','official')['data']['rows'])
    assert any(r['id']=='V000412' for r in flux.call('task','image','versions','--image-id','BJX00000001')['data'])
    project=flux.call('project','info','--project-id',parent['project_id'])['data']
    assert int(project['userId'])==4669
    directory='/workspace/amp_reo/logs/x1_amp_exp29/exported_data/exp2.9_long_r1'
    command=('gm-run amp_reo/humanoid/scripts/resume_amp.py --task=x1_amp_exp29 --headless --num_envs=4096'
        ' --seed=5 --max_iterations=5000 --run_name=exp2.9_long'
        ' --recovery_model='+directory+'/input_model_8500.pt --recovery_sha256='+digest+
        ' --source_iteration=8500 --target_iteration=13500')
    state=dict(experiment='exp2.9_long',account=ACCOUNT,task='x1_amp_exp29',branch=BRANCH,
        commit=commit,parent_algorithm_commit=PARENT_COMMIT,project_id=parent['project_id'],original_task_id=parent['task_id'],
        training_mode='anchored_continuation_preserve_critic_optimizers',resume_mode='full_checkpoint',
        source_iteration=8500,new_initial_iteration=8500,target_iteration=13500,max_iterations=5000,
        recovery_additional_updates=5000,additional_updates=5000,num_envs=4096,seed=5,
        source_checkpoint=model.relative_to(ROOT).as_posix(),source_checkpoint_sha256=digest,
        policy_origin_source_sha256=checkpoint['training_origin']['source_sha256'],
        observation_dim=6468,action_dim=29,checkpoint_dir=directory,start_command=command,
        first_checkpoint_iteration=8600,max_wall_seconds=19800,source_discriminator_updates=8300,
        current_goods_price_per_hour=goods['goodsPrice'],phase='prepared',
        source_training_origin=checkpoint['training_origin'],simulator_and_rng_restored=False,
        single_task_sequential=True,created_at=datetime.now(timezone.utc).isoformat())
    if replacement:state['replaces_failed_task_id']=replacement
    write(DIRECTORY/'plan.json',state)
    encoded=base64.urlsafe_b64encode(row['policUrlDown'].encode()).decode()
    payload=dict(taskBaseInfo=dict(projectId=parent['project_id'],taskType='1',trainType='1',taskName=NAME,
        taskDescription='After exp2.9 completes8500, exact full checkpoint continuation +5000 to13500; '
            'preserve policy,critic,D8300,normalizer,bothAdam,anchor,curriculum andallalgorithm/PD/phase settings. '
            '4096 seed5 24rollout2x4;save100,best;regression protection and5.5h limit.',
        taskTag=['exp2.9','overnight','full-resume'],goodsId='ESKU000001',imageId='BJX00000001',imageVersion='V000412',personalDataPath=''),
        taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/amp_reo.git',
            versionType='1',versionName=BRANCH)]),mainCodeUri='amp_reo/humanoid/scripts/resume_amp.py',
            hparamsPath='amp_reo/humanoid/envs/x1/x1_amp_exp29_config.py',startScript=command+' --recovery_url_b64='+encoded,isOpen='0'),
        runtimeReminderConfig=dict(enableRuntimeReminder=False,reminderDurations=[]))
    task_id=create('task',payload)
    state.update(task_id=task_id,phase='created');write(STATE,state)
    print('LONG_TASK_CREATED '+task_id,flush=True)
    assert flux.call('--dry-run','task','run','--task-id',task_id)['success']
    flux.call('task','run','--task-id',task_id)
    state.update(phase='submitted',submitted_at=datetime.now(timezone.utc).isoformat());write(STATE,state)
    print(json.dumps({k:state[k] for k in ('task_id','phase','source_iteration','target_iteration','account')}),flush=True)


if __name__=='__main__':main()
