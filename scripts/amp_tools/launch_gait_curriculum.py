"""Create one authorized1000-update exp1.15 task; never duplicate existing state/tasks."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import flux_ops as flux
from launch_reference_route import identifiers
from private_training_input import request as private_request

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 'limx2f1e1sojm8yk@maxxspace.com'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', choices=('115', '115t'), required=True)
    p.add_argument('--replace-failed', action='store_true', help='Replace a diagnosed terminal zero-update teacher attempt; never duplicate a live task')
    a = p.parse_args()
    experiment = 'exp1.15'+('t' if a.route.endswith('t') else '')
    task = 'x1_amp_exp'+a.route
    records = ROOT/'czy/exp1/records'
    state_path = records/(experiment+'_training.json')
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    if previous and not a.replace_failed: raise ValueError('Training state already exists; inspect this task instead of creating again')
    if a.replace_failed:
        assert previous and a.route == '115t' and previous['account'] == ACCOUNT
    directory = records/(experiment+'_evaluation')
    plan = json.loads((directory/'plan.json').read_text())
    publication = json.loads((directory/'publication.json').read_text())
    assert publication['published'] and publication['production_checkpoint_preflight']
    assert plan['training_account'] == ACCOUNT and plan['additional_updates'] == 1000
    assert hashlib.sha256((ROOT/plan['source_model']).read_bytes()).hexdigest() == plan['source_sha256']
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    remote = subprocess.check_output(['git', '-c', 'http.proxy='+publication['proxy'], '-c', 'http.sslBackend=openssl',
                                     'ls-remote', '--heads', 'origin', publication['branch']], env=env).decode().split()[0]
    assert remote == publication['commit']
    flux.ACCOUNT = ACCOUNT
    who = flux.call('auth', 'whoami')['data']
    assert who['email'] == ACCOUNT and int(who['userId']) == 4684
    project = flux.call('project', 'info', '--project-id', plan['project_id'])['data']
    assert int(project['userId']) == 4684
    existing = flux.call('task', 'list', '--project-id', plan['project_id'], '--task-name', experiment+'-', '--limit', '50')['data']['rows']
    teacher_runtime_url = None
    failed_attempt = None
    if a.replace_failed:
        assert {r['taskId'] for r in existing} == {previous['task_id']}, 'An additional platform task exists; inspect before replacement'
        failed = flux.call('task', 'info', '--task-id', previous['task_id'])['data']
        assert str(failed['taskBaseInfo']['taskStatus']) == '6' and str(failed['taskBaseInfo']['userId']) == '4684'
        log_env = os.environ.copy(); log_env.update(FLUX_API_KEY=flux.credentials()['api_key'],
            FLUX_BASE_URL=os.environ.get('FLUX_LOCAL_BASE_URL', 'https://internal.limxdynamics.com/dev-api'))
        log = subprocess.check_output([str(flux.CLI), '--yes', 'task', 'logs', '--task-id', previous['task_id'],
                                       '--raw', '--no-request-log'], env=log_env).decode('utf-8', 'replace')
        assert 'FileNotFoundError' in log and 'Learning iteration' not in log
        teacher_runtime_url = failed['taskCodeInfo']['checkPointFilePathSignedUrl']
        assert teacher_runtime_url.startswith('https://')
        status, _, blob = private_request(teacher_runtime_url)
        assert status == 200 and hashlib.sha256(blob).hexdigest() == plan['teacher_sha256']
        failed_attempt = dict(task_id=previous['task_id'], commit=previous['commit'], learning_updates=0,
            runtime_seconds=failed['taskBaseInfo'].get('runtime'), started_at=failed['taskBaseInfo'].get('startTime'),
            ended_at=failed['taskBaseInfo'].get('endTime'), reason='Official teacher checkpoint mount did not materialize its file',
            terminal_status='6', replacement_authorized=True)
    else:
        assert not existing, 'Platform task already exists; inspect it before any creation'
    resources = flux.call('task', 'resource', 'list', '--goods-back-category', '3', '--limit', '20')['data']['rows']
    images = flux.call('task', 'image', 'official')['data']['rows']
    versions = flux.call('task', 'image', 'versions', '--image-id', 'BJX00000001')['data']
    assert any(r['goodsId'] == 'ESKU000001' for r in resources)
    assert any(r['imageId'] == 'BJX00000001' for r in images)
    assert any(r['id'] == 'V000412' for r in versions)
    flux.ACCOUNT = plan['source_account']
    source_info = flux.call('task', 'info', '--task-id', plan['source_task_id'])['data']
    assert int(source_info['taskBaseInfo']['userId']) == 4681
    rows = flux.call('task', 'model', 'list', '--task-id', plan['source_task_id'], '--checkpoint', '5300', '--limit', '30')['data']['rows']
    source = next(r for r in rows if r['fileName'] == 'model_5300.pt' and str(r['checkpoint']) == '5300')
    encoded = base64.urlsafe_b64encode(source['policUrlDown'].encode()).decode()
    flux.ACCOUNT = ACCOUNT
    teacher_input = None
    if a.route.endswith('t'):
        receipt = json.loads((directory/'teacher_input.json').read_text())
        assert receipt['verified'] and receipt['account'] == ACCOUNT and receipt['sha256'] == plan['teacher_sha256']
        assert receipt['official_personal_storage'] and receipt['download_sha256_verified']
        teacher_input = receipt['storage_path'].split('/')[-1]
    checkpoint_dir = '/workspace/amp_reo/logs/'+task+'/exported_data/'+experiment+('_runtime' if a.replace_failed else '_main')
    entry = 'train_guided_amp.py' if teacher_input else 'train.py'
    command = ('gm-run amp_reo/humanoid/scripts/'+entry+' --task='+task+
               ' --headless --num_envs=4096 --seed=5 --max_iterations=1000 --run_name='+experiment+
               ' --checkpoint_dir='+checkpoint_dir+' --warm_start='+checkpoint_dir+
               '/input_exp114_model_5300.pt --warm_start_sha256='+plan['source_sha256'])
    if teacher_input and not teacher_runtime_url: command += ' --teacher_path='+teacher_input
    runtime_command = command+' --warm_start_url_b64='+encoded
    if teacher_runtime_url: runtime_command += ' --teacher_url_b64='+base64.urlsafe_b64encode(teacher_runtime_url.encode()).decode()
    payload = {'taskBaseInfo': {'projectId':plan['project_id'], 'taskType':'1', 'trainType':'1',
        'taskName':experiment+'-normalized-forward-'+('teacher' if teacher_input else 'DH')+'-4096-seed5-plus1000'+('-runtime' if a.replace_failed else ''),
        'taskDescription':'Stage1:80%+0.3m/s/20%zero; normalized phase q/dq/feet, temporal amplitude/support/net displacement; '
                          'AMP reward off500 then ramp to1500; stable5300 DH source; fresh critic/D/Adam; '+
                          ('exact exp1.9 forward MLP action transplant; ' if teacher_input else '')+
                          '1000updates,24rollout,2x4PPO,D4. Evaluate before any range/budget extension.',
        'taskTag':[experiment, 'AMP', 'forward-curriculum'], 'goodsId':'ESKU000001', 'imageId':'BJX00000001',
        'imageVersion':'V000412', 'personalDataPath':'/personal/' if teacher_input and not teacher_runtime_url else ''},
        'taskCodeInfo': {'codeType':'2', 'codeUrl':json.dumps([{'codeUrl':'https://github.com/Lee-Weather/amp_reo.git',
            'versionType':'1', 'versionName':publication['branch']}]), 'mainCodeUri':'amp_reo/humanoid/scripts/'+entry,
            'hparamsPath':'amp_reo/humanoid/envs/x1/'+task+'_config.py', 'startScript':runtime_command, 'isOpen':'0'},
        'runtimeReminderConfig':{'enableRuntimeReminder':False, 'reminderDurations':[]}}
    state = dict(experiment=experiment, account=ACCOUNT, account_identity_verified={'email':ACCOUNT, 'user_id':4684},
        task=task, branch=publication['branch'], commit=publication['commit'], project_id=plan['project_id'],
        source_task_id=plan['source_task_id'], source_checkpoint=plan['source_model'], source_iteration=5300,
        source_checkpoint_sha256=plan['source_sha256'], training_mode='policy_transfer_fresh_critic_amp_optimizers',
        new_initial_iteration=0, max_iterations=1000, target_iteration=1000, num_envs=4096, seed=5,
        num_steps_per_env=24, num_learning_epochs=2, num_mini_batches=4, discriminator_updates_per_iteration=4,
        sample_budget=98304000, observation_dim=6468, action_dim=29, checkpoint_dir=checkpoint_dir,
        start_command=command, first_checkpoint_iteration=100, goods_id='ESKU000001', image_id='BJX00000001',
        image_version='V000412', phase='validated_not_created')
    if teacher_input: state.update(teacher_sha256=plan['teacher_sha256'], teacher_source_iteration=11250,
                                    teacher_storage_path=receipt['storage_path'], teacher_method='exact_MLP_to_DH_action_function')
    if failed_attempt:
        state.update(failed_startup_attempts=[failed_attempt], replaces_failed_task=previous['task_id'],
            teacher_transport='Official private signed URL approved by user; in-memory download with SHA check',
            private_url_argument_user_approved=True, signed_urls_persisted_locally=False)
        (directory/'failed_startup.json').write_text(json.dumps(failed_attempt, indent=2)+'\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='.flux-gait-', dir=ROOT) as temp:
        request = Path(temp)/'create.json'
        request.write_text(json.dumps(payload), encoding='utf-8')
        relative = request.relative_to(ROOT).as_posix()
        assert flux.call('--dry-run', 'task', 'create', '--file', relative)['success']
        created = flux.call('task', 'create', '--file', relative)
        request.unlink()
        ids = set(identifiers(created['data']))
        if len(ids) != 1: raise RuntimeError('Creation succeeded; inspect task ID, do not create again')
        state.update(task_id=ids.pop(), phase='created', created_at=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        print('TASK_CREATED '+state['task_id'], flush=True)
    assert flux.call('--dry-run', 'task', 'run', '--task-id', state['task_id'])['success']
    flux.call('task', 'run', '--task-id', state['task_id'])
    state.update(phase='submitted', submitted_at=datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
