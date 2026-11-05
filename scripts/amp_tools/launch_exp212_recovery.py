"""User-approved recovery from the verified stopped13550 checkpoint; target14500."""
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
STATE=ROOT/'czy/exp1/records/exp2.12_training.json'
EV=ROOT/'czy/exp1/records/exp2.12_evaluation'
NAME='exp2.12-resetfix-resume13550-to14500-plus950'
BRANCH='codex/exp2-12-reset-fix'


def main():
    old=json.loads(STATE.read_text());flux.ACCOUNT=old['account']
    who=flux.call('auth','whoami')['data'];assert who['email']==old['account'] and int(who['userId'])==4670
    if old.get('refinement')=='exp212_nominal_reset_fix_v1':
        print(json.dumps(dict(task_id=old['task_id'],existing_task=True)));return
    assert old['task_id']=='TASK_20261104_184'
    source_info=flux.call('task','info','--task-id',old['task_id'])['data']
    assert str(source_info['taskBaseInfo']['taskStatus'])=='6' and int(source_info['taskBaseInfo']['userId'])==4670
    assert not flux.call('task','list','--task-name',NAME,'--limit','50')['data']['rows'],'Inspect existing creation; never duplicate'
    proof=json.loads((EV/'reset_fix/recovery_source_verification.json').read_text())
    assert proof['source_iteration']==13550 and proof['remaining_updates']==950 and proof['saved_and_platform_download_verified']
    digest=proof['source_sha256'];model=ROOT/proof['source_checkpoint'];assert hashlib.sha256(model.read_bytes()).hexdigest()==digest
    rows=flux.call('task','model','list','--task-id',old['task_id'],'--checkpoint','13550','--limit','30')['data']['rows']
    row=next(r for r in rows if r['fileName']=='model_13550.pt' and str(r['checkpoint'])=='13550')
    status,headers,raw=request(row['policUrlDown']);assert status==200 and hashlib.sha256(raw).hexdigest()==digest
    assert len(raw)==int(headers['content-length']) and hashlib.md5(raw).hexdigest()==headers['etag'].strip('"').lower()
    encoded=base64.urlsafe_b64encode(row['policUrlDown'].encode()).decode();del raw
    publication=json.loads((EV/'reset_fix/publication.json').read_text());assert publication['published'] and publication['exact_isolated_tree_tests']['passed']
    commit=publication['remote_commit']
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    remote=subprocess.check_output(['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl','ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    goods=next(r for r in flux.call('task','resource','list','--goods-back-category','3','--limit','20')['data']['rows'] if r['goodsId']=='ESKU000001')
    assert goods['goodsBackSku']['gpuNum']==1 and goods['goodsBackSku']['gpuModel']=='4090D'
    assert any(r['imageId']=='BJX00000001' for r in flux.call('task','image','official')['data']['rows'])
    assert any(r['id']=='V000412' for r in flux.call('task','image','versions','--image-id','BJX00000001')['data'])
    project=flux.call('project','info','--project-id',old['project_id'])['data'];assert int(project['userId'])==4670
    directory=old['checkpoint_dir']
    command=('gm-run amp_reo/humanoid/scripts/train_nominal_amp.py --task=x1_amp_exp212 --headless'
        ' --num_envs=4096 --seed=5 --max_iterations=950 --run_name=exp2.12 --recovery_model='+directory+
        '/input_model_13550.pt --recovery_sha256='+digest+' --source_iteration=13550 --target_iteration=14500')
    state={k:v for k,v in old.items() if k not in ('task_id','first_checkpoint_upload_verification','first_checkpoint_inspection','submitted_at','status','latest_log_lines')}
    state.update(commit=commit,branch=BRANCH,parent_algorithm_commit=publication['parent'],
        source_account=old['account'],source_task_id=old['task_id'],source_checkpoint=proof['source_checkpoint'],
        source_checkpoint_sha256=digest,source_iteration=13550,new_initial_iteration=13550,
        max_iterations=950,additional_updates=950,recovery_additional_updates=950,source_discriminator_updates=13350,
        first_checkpoint_iteration=13600,refinement='exp212_nominal_reset_fix_v1',
        recovered_from_task_id=old['task_id'],original_experiment_source_iteration=13500,
        original_experiment_target_iteration=14500,original_experiment_update_budget=1000,
        start_command=command,sample_budget=950*4096*24,current_goods_price_per_hour=goods['goodsPrice'],
        phase='prepared_recovery',created_at=datetime.now(timezone.utc).isoformat(),
        partial_source_limitation='First50 retained updates had noise/friction/forward cohort but reset hook was bypassed; corrected reset applies to remaining950.',
        discarded_unsaved_or_unuploaded_updates=proof['unsaved_or_unuploaded_updates_not_resumed'])
    write(EV/'reset_fix/recovery_plan.json',state)
    payload=dict(taskBaseInfo=dict(projectId=old['project_id'],taskType='1',trainType='1',taskName=NAME,
        taskDescription='User-approved reset-entry fix; original task184 stopped. Recover verified13550, exactly950 additional updates to original14500 target. Retain bothAdam,D13350,normalizer,anchor.25pct exact standing/no-noise/forward cohort,75pct original mix;original4096seed5 PD rewards phase and guards.',
        taskTag=['exp2.12','AMP','recovery'],goodsId='ESKU000001',imageId='BJX00000001',imageVersion='V000412',personalDataPath=''),
        taskCodeInfo=dict(codeType='2',codeUrl=json.dumps([dict(codeUrl='https://github.com/Lee-Weather/amp_reo.git',versionType='1',versionName=BRANCH)]),
        mainCodeUri='amp_reo/humanoid/scripts/train_nominal_amp.py',hparamsPath='amp_reo/humanoid/envs/x1/x1_amp_exp212_config.py',
        startScript=command+' --recovery_url_b64='+encoded,isOpen='0'),runtimeReminderConfig=dict(enableRuntimeReminder=False,reminderDurations=[]))
    task_id=create('task',payload)
    archive=EV/'reset_fix/stopped_training_state.json';assert not archive.exists();write(archive,old)
    state.update(task_id=task_id,phase='created_recovery');write(STATE,state)
    print('RECOVERY_TASK_CREATED '+task_id,flush=True)
    assert flux.call('--dry-run','task','run','--task-id',task_id)['success']
    flux.call('task','run','--task-id',task_id)
    state.update(phase='submitted_recovery',submitted_at=datetime.now(timezone.utc).isoformat());write(STATE,state)
    print(json.dumps({k:state[k] for k in ('task_id','account','commit','source_iteration','target_iteration','additional_updates')}),flush=True)


if __name__=='__main__':main()
