"""Create and start exp1.18: continue exp1.17 model7000 for 5000 more updates."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import flux_ops as flux

ROOT = Path(__file__).resolve().parents[2]
TARGET_ACCOUNT = 'limx2f205wu6b0jq@maxxspace.com'
SOURCE_ACCOUNT = 'limx2f1e1sojm8yk@maxxspace.com'
PROJECT_ID = 'PRO_20261031_006'
SOURCE_TASK = 'TASK_20261031_027'
SOURCE_SHA = '4edb874efe478baea76af598db86e923c24d62e1653542bf144111a453971ac8'
BRANCH = 'codex/exp1-18'
PROXY = 'socks5h://10.12.201.122:39009'


def main():
    records = ROOT/'czy/exp1/records'
    directory = records/'exp1.18_evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    state_path = records/'exp1.18_training.json'
    if state_path.exists():
        raise ValueError('Recorded exp1.18 state exists; inspect it instead of submitting again')
    publication = json.loads((directory/'publication.json').read_text())
    assert publication['published'] and publication['parent'] == '57cfd3cc04169c3d6004d3a924ba9ed6269659d4'
    commit = publication['remote_commit']
    assert hashlib.sha256((ROOT/'czy/data/exp1.17/model_7000.pt').read_bytes()).hexdigest() == SOURCE_SHA
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    remote = subprocess.check_output(['git', '-c', 'http.proxy='+PROXY, '-c', 'http.sslBackend=openssl',
        'ls-remote', '--heads', 'origin', BRANCH], env=env).decode().split()[0]
    assert remote == commit

    # Source model URL is read with the source account and kept in memory only.
    flux.ACCOUNT = SOURCE_ACCOUNT
    who = flux.call('auth', 'whoami')['data']
    assert who['email'] == SOURCE_ACCOUNT and int(who['userId']) == 4684
    info = flux.call('task', 'info', '--task-id', SOURCE_TASK)['data']
    assert int(info['taskBaseInfo']['userId']) == 4684
    rows = flux.call('task', 'model', 'list', '--task-id', SOURCE_TASK, '--checkpoint', '7000',
                     '--limit', '30')['data']['rows']
    source = next(r for r in rows if r['fileName'] == 'model_7000.pt' and str(r['checkpoint']) == '7000')
    encoded = base64.urlsafe_b64encode(source['policUrlDown'].encode()).decode()

    flux.ACCOUNT = TARGET_ACCOUNT
    who = flux.call('auth', 'whoami')['data']
    assert who['email'] == TARGET_ACCOUNT and int(who['userId']) == 4685
    project = flux.call('project', 'info', '--project-id', PROJECT_ID)['data']
    assert int(project['userId']) == 4685 and project['projectName'] == 'X1 29DOF AMP exp1.18 continuation'
    rows = flux.call('task', 'list', '--project-id', PROJECT_ID, '--task-name', 'exp1.18-', '--limit', '50')['data']['rows']
    assert not rows, 'A platform task exists; inspect before creating again'
    resources = flux.call('task', 'resource', 'list', '--goods-back-category', '3', '--limit', '20')['data']['rows']
    images = flux.call('task', 'image', 'official')['data']['rows']
    versions = flux.call('task', 'image', 'versions', '--image-id', 'BJX00000001')['data']
    assert any(r['goodsId'] == 'ESKU000001' for r in resources)
    assert any(r['imageId'] == 'BJX00000001' for r in images)
    assert any(r['id'] == 'V000412' for r in versions)

    checkpoint_dir = '/workspace/amp_reo/logs/x1_amp_exp118/exported_data/exp1.18_main'
    command = ('gm-run amp_reo/humanoid/scripts/train.py --task=x1_amp_exp118 --headless --num_envs=4096'
               ' --seed=5 --max_iterations=5000 --run_name=exp1.18 --checkpoint_dir='+checkpoint_dir+
               ' --warm_start='+checkpoint_dir+'/input_exp117_model_7000.pt --warm_start_sha256='+SOURCE_SHA)
    plan = dict(experiment='exp1.18', task='x1_amp_exp118', account=TARGET_ACCOUNT, project_id=PROJECT_ID,
        branch=BRANCH, commit=commit, source_experiment='exp1.17', source_task_id=SOURCE_TASK,
        source_checkpoint='czy/data/exp1.17/model_7000.pt', source_sha256=SOURCE_SHA, source_iteration=7000,
        additional_updates=5000, target_iteration=12000, num_envs=4096, seed=5, num_steps_per_env=24,
        num_learning_epochs=2, num_mini_batches=4, discriminator_updates_per_iteration=4,
        sample_budget=5000*4096*24, observation_dim=6468, action_dim=29,
        training_mode='reward_refinement_preserve_policy_amp', first_checkpoint_iteration=7100,
        status='submitting', created_at=datetime.now(timezone.utc).isoformat())
    (directory/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')

    payload = {'taskBaseInfo': {'projectId': PROJECT_ID, 'taskType': '1', 'trainType': '1',
        'taskName': 'exp1.18-continue-4096-seed5-plus5000',
        'taskDescription': 'Continue exp1.17 model7000 to12000: unchanged absolute phase/support rewards; '
            'retain actor/CNN/estimator/logstd/D/normalizer/D Adam; fresh critic/PPO; fixed AMP0.5; '
            '5000 new updates,24rollout,2x4PPO,D4.',
        'taskTag': ['exp1.18', 'AMP', 'continuation'], 'goodsId': 'ESKU000001', 'imageId': 'BJX00000001',
        'imageVersion': 'V000412', 'personalDataPath': ''},
        'taskCodeInfo': {'codeType': '2', 'codeUrl': json.dumps([{'codeUrl': 'https://github.com/Lee-Weather/amp_reo.git',
            'versionType': '1', 'versionName': BRANCH}]), 'mainCodeUri': 'amp_reo/humanoid/scripts/train.py',
            'hparamsPath': 'amp_reo/humanoid/envs/x1/x1_amp_exp118_config.py',
            'startScript': command+' --warm_start_url_b64='+encoded, 'isOpen': '0'},
        'runtimeReminderConfig': {'enableRuntimeReminder': False, 'reminderDurations': []}}
    state = dict(experiment='exp1.18', account=TARGET_ACCOUNT, account_identity_verified={'email': TARGET_ACCOUNT, 'user_id': 4685},
        task='x1_amp_exp118', branch=BRANCH, commit=commit, project_id=PROJECT_ID,
        source_task_id=SOURCE_TASK, source_checkpoint='czy/data/exp1.17/model_7000.pt', source_iteration=7000,
        source_checkpoint_sha256=SOURCE_SHA, training_mode='reward_refinement_preserve_policy_amp',
        new_initial_iteration=7000, max_iterations=5000, target_iteration=12000, num_envs=4096, seed=5,
        num_steps_per_env=24, num_learning_epochs=2, num_mini_batches=4, discriminator_updates_per_iteration=4,
        sample_budget=5000*4096*24, observation_dim=6468, action_dim=29, checkpoint_dir=checkpoint_dir,
        start_command=command, first_checkpoint_iteration=7100, goods_id='ESKU000001', image_id='BJX00000001',
        image_version='V000412', phase='validated_not_created', signed_urls_persisted_locally=False)
    with tempfile.TemporaryDirectory(prefix='.flux-exp118-', dir=ROOT) as temp:
        request = Path(temp)/'create.json'; request.write_text(json.dumps(payload), encoding='utf-8')
        relative = request.relative_to(ROOT).as_posix()
        assert flux.call('--dry-run', 'task', 'create', '--file', relative)['success']
        created = flux.call('task', 'create', '--file', relative)
        request.unlink()
        data = created['data']
        task_id = data if isinstance(data, str) else (data.get('taskId') or data.get('id'))
        assert isinstance(task_id, str) and task_id, 'Creation returned; inspect platform before another creation'
        state.update(task_id=task_id, phase='created', created_at=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        print('TASK_CREATED '+task_id, flush=True)
    assert flux.call('--dry-run', 'task', 'run', '--task-id', task_id)['success']
    flux.call('task', 'run', '--task-id', task_id)
    state.update(phase='submitted', submitted_at=datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    plan.update(status='submitted', task_id=task_id)
    (directory/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: state[k] for k in ('experiment', 'task_id', 'phase', 'account', 'project_id', 'commit')}), flush=True)


if __name__ == '__main__':
    main()
