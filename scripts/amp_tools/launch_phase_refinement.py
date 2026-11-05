"""Submit exp1.16 once, using the private exp1.15t model as a verified input."""
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

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 'limx2f1e1sojm8yk@maxxspace.com'


def main():
    records = ROOT/'czy/exp1/records'
    directory = records/'exp1.16_evaluation'
    state_path = records/'exp1.16_training.json'
    assert not state_path.exists(), 'Inspect the recorded task; never submit a duplicate'
    plan = json.loads((directory/'plan.json').read_text())
    publication = json.loads((directory/'publication.json').read_text())
    assert publication['published'] and publication['production_checkpoint_preflight']
    assert (plan['additional_updates'], plan['target_iteration'], plan['planned_training_account']) == (3000, 4000, ACCOUNT)
    assert hashlib.sha256((ROOT/plan['source_checkpoint']).read_bytes()).hexdigest() == plan['source_sha256']
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    remote = subprocess.check_output(['git', '-c', 'http.proxy='+publication['proxy'], '-c', 'http.sslBackend=openssl',
        'ls-remote', '--heads', 'origin', publication['branch']], env=env).decode().split()[0]
    assert remote == publication['commit']
    flux.ACCOUNT = ACCOUNT
    who = flux.call('auth', 'whoami')['data']
    assert who['email'] == ACCOUNT and int(who['userId']) == 4684
    assert int(flux.call('project', 'info', '--project-id', plan['project_id'])['data']['userId']) == 4684
    rows = flux.call('task', 'list', '--project-id', plan['project_id'], '--task-name', 'exp1.16-', '--limit', '50')['data']['rows']
    assert not rows, 'A platform task exists; inspect before creating again'
    resources = flux.call('task', 'resource', 'list', '--goods-back-category', '3', '--limit', '20')['data']['rows']
    images = flux.call('task', 'image', 'official')['data']['rows']
    versions = flux.call('task', 'image', 'versions', '--image-id', 'BJX00000001')['data']
    assert any(r['goodsId'] == 'ESKU000001' for r in resources)
    assert any(r['imageId'] == 'BJX00000001' for r in images)
    assert any(r['id'] == 'V000412' for r in versions)
    info = flux.call('task', 'info', '--task-id', plan['source_task_id'])['data']
    assert int(info['taskBaseInfo']['userId']) == 4684
    rows = flux.call('task', 'model', 'list', '--task-id', plan['source_task_id'], '--checkpoint', '1000', '--limit', '30')['data']['rows']
    source = next(r for r in rows if r['fileName'] == 'model_1000.pt' and str(r['checkpoint']) == '1000')
    encoded = base64.urlsafe_b64encode(source['policUrlDown'].encode()).decode()
    checkpoint_dir = '/workspace/amp_reo/logs/x1_amp_exp116/exported_data/exp1.16_main'
    command = ('gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp_exp116 --headless --num_envs=4096'
               ' --seed=5 --max_iterations=3000 --run_name=exp1.16 --checkpoint_dir='+checkpoint_dir+
               ' --warm_start='+checkpoint_dir+'/input_exp115t_model_1000.pt --warm_start_sha256='+plan['source_sha256'])
    payload = {'taskBaseInfo': {'projectId':plan['project_id'], 'taskType':'1', 'trainType':'1',
        'taskName':'exp1.16-phase-refinement-retainAMP-4096-seed5-plus3000',
        'taskDescription':'Continue exp1.15t1000 to4000: signed dense phase progress and soft support; '
            'retain actor/CNN/estimator/logstd/D/normalizer/D Adam; fresh critic/PPO; fixedAMP0.5; '
            '80%+0.3m/s20%zero; 100-update LR protection; 3000 new updates,24rollout,2x4PPO,D4.',
        'taskTag':['exp1.16', 'AMP', 'phase-refinement'], 'goodsId':'ESKU000001', 'imageId':'BJX00000001',
        'imageVersion':'V000412', 'personalDataPath':''},
        'taskCodeInfo': {'codeType':'2', 'codeUrl':json.dumps([{'codeUrl':'https://github.com/Lee-Weather/amp_reo.git',
            'versionType':'1', 'versionName':publication['branch']}]), 'mainCodeUri':'amp_reo/humanoid/scripts/train.py',
            'hparamsPath':'amp_reo/humanoid/envs/x1/x1_amp_exp116_config.py',
            'startScript':command+' --warm_start_url_b64='+encoded, 'isOpen':'0'},
        'runtimeReminderConfig':{'enableRuntimeReminder':False, 'reminderDurations':[]}}
    state = dict(experiment='exp1.16', account=ACCOUNT, account_identity_verified={'email':ACCOUNT, 'user_id':4684},
        task='x1_amp_exp116', branch=publication['branch'], commit=publication['commit'], project_id=plan['project_id'],
        source_task_id=plan['source_task_id'], source_checkpoint=plan['source_checkpoint'], source_iteration=1000,
        source_checkpoint_sha256=plan['source_sha256'], training_mode=plan['training_mode'], new_initial_iteration=1000,
        max_iterations=3000, target_iteration=4000, num_envs=4096, seed=5, num_steps_per_env=24,
        num_learning_epochs=2, num_mini_batches=4, discriminator_updates_per_iteration=4, sample_budget=294912000,
        observation_dim=6468, action_dim=29, checkpoint_dir=checkpoint_dir, start_command=command,
        first_checkpoint_iteration=1100, goods_id='ESKU000001', image_id='BJX00000001', image_version='V000412',
        phase='validated_not_created', signed_urls_persisted_locally=False)
    with tempfile.TemporaryDirectory(prefix='.flux-phase-', dir=ROOT) as temp:
        request = Path(temp)/'create.json'; request.write_text(json.dumps(payload), encoding='utf-8')
        relative = request.relative_to(ROOT).as_posix()
        assert flux.call('--dry-run', 'task', 'create', '--file', relative)['success']
        created = flux.call('task', 'create', '--file', relative)
        request.unlink()
        ids = set(identifiers(created['data']))
        assert len(ids) == 1, 'Creation returned; inspect platform before another creation'
        state.update(task_id=ids.pop(), phase='created', created_at=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        print('TASK_CREATED '+state['task_id'], flush=True)
    assert flux.call('--dry-run', 'task', 'run', '--task-id', state['task_id'])['success']
    flux.call('task', 'run', '--task-id', state['task_id'])
    state.update(phase='submitted', submitted_at=datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
