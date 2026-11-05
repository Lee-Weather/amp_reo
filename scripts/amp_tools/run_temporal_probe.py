"""Launch/collect the frozen exploration diagnostic in a fresh SSH workspace."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from launch_server_amp import HOST, SCP, remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    launch = sub.add_parser('launch')
    launch.add_argument('--training-state', type=Path, required=True)
    launch.add_argument('--job', type=Path, required=True)
    launch.add_argument('--kind', choices=('temporal', 'advantages'), default='temporal')
    collect = sub.add_parser('collect')
    collect.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'collect':
        state = json.loads(args.job.read_text())
        program = '''
import pathlib,json,hashlib
s=json.loads(__STATE__);root=pathlib.Path(s['server_dir']);proc=pathlib.Path('/proc/%s/stat'%s['launch_pid'])
alive=proc.exists() and proc.read_text().split()[2]!='Z'
log=(root/'probe.log').read_text();complete=s.get('success_marker','TEMPORAL_PROBE_OK') in log
result=dict(alive=alive,complete=complete,log_tail=log.splitlines()[-4:])
if not alive and complete:
    report=root/s.get('report_name','temporal_probe_summary.json');result['report']=json.loads(report.read_text())
    assert result['report']['checkpoint_sha256']==s['checkpoint_sha256']
    assert hashlib.sha256((root/s.get('script_name','probe_temporal_exploration.py')).read_bytes()).hexdigest()==s['probe_sha256']
print(json.dumps(result))
'''.replace('__STATE__', repr(json.dumps(state)))
        result = json.loads(remote('python3 -', program))
        if not result['alive'] and result['complete']:
            report = result['report']
            report.update(server_dir=state['server_dir'], code_sha256=state['code_sha256'], probe_sha256=state['probe_sha256'])
            args.job.with_name(state.get('report_name','temporal_probe_summary.json')).write_text(json.dumps(report, indent=2)+'\n')
            state['phase'] = 'completed_and_analyzed'
            args.job.write_text(json.dumps(state, indent=2)+'\n')
            print(json.dumps(report.get('cases',report.get('groups'))))
        else:
            print(json.dumps(result))
        return
    if args.job.exists():
        raise FileExistsError('Inspect the existing job process, never launch a duplicate')
    training = json.loads(args.training_state.read_text())
    assert training['host'] == HOST and training['task'] == 'x1_amp_exp18'
    final = training['target_iteration']
    assert final == 11000
    receipt = next(r for r in training['status']['saved_checkpoints'] if r['iteration'] == final)
    assert str(final) in training['verified_checkpoints']
    assert receipt['path'].startswith(training['server_dir']+'/')
    if remote('nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip():
        raise RuntimeError('GPU busy; inspect current workloads before launching')
    script_name = 'probe_temporal_exploration.py' if args.kind == 'temporal' else 'audit_directional_advantages.py'
    report_name = 'temporal_probe_summary.json' if args.kind == 'temporal' else 'directional_advantage_summary.json'
    success_marker = 'TEMPORAL_PROBE_OK' if args.kind == 'temporal' else 'DIRECTIONAL_ADVANTAGE_AUDIT_OK'
    script = Path(__file__).with_name(script_name)
    server = '/home/robot/czy/exp1/exp_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    state = dict(purpose='Frozen GPU exploration diagnosis, no training or acceptance', server_dir=server,
                 training_dir=training['server_dir'], checkpoint_iteration=final,
                 checkpoint_sha256=receipt['sha256'], checkpoint_source=receipt['path'],
                 code_sha256=training['code_sha256'], probe_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                 num_envs=4096, seed=5, phase='preparing', launch_pid=None,
                 script_name=script_name, report_name=report_name, success_marker=success_marker)
    args.job.parent.mkdir(parents=True, exist_ok=True)
    args.job.write_text(json.dumps(state, indent=2)+'\n')
    program = '''
import pathlib,shutil,hashlib,json
s=json.loads(__STATE__);root=pathlib.Path(s['server_dir']);root.mkdir(exist_ok=False)
source=pathlib.Path(s['training_dir'])
shutil.copytree(source/'humanoid',root/'humanoid',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(source/'setup.py',root/'setup.py');(root/'resources').symlink_to(source/'resources',target_is_directory=True)
h=hashlib.sha256()
for p in sorted((root/'humanoid').rglob('*.py'))+[root/'setup.py']:
    h.update(p.relative_to(root).as_posix().encode()+b'\\0'+p.read_bytes()+b'\\0')
assert h.hexdigest()==s['code_sha256']
model=root/'model_11000.pt';shutil.copy2(s['checkpoint_source'],model)
assert hashlib.sha256(model.read_bytes()).hexdigest()==s['checkpoint_sha256']
print('TEMPORAL_PROBE_PREPARED')
'''.replace('__STATE__', repr(json.dumps(state)))
    remote('python3 -', program)
    subprocess.run([SCP, '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no', str(script),
                    HOST+':'+server+'/'+script_name], check=True, timeout=60)
    command = ['timeout', '300', 'python', '-u', script_name, '--task=x1_amp_exp18',
               '--headless', '--num_envs=4096', '--seed=5', '--model=model_11000.pt',
               '--sha256='+receipt['sha256'], '--output='+report_name]
    shell = '\n'.join(['set -euo pipefail', 'cd '+shlex.quote(server),
                       'source /home/robot/Anaconda/etc/profile.d/conda.sh', 'conda activate F1',
                       'export PYTHONPATH='+shlex.quote(server), 'export OMP_NUM_THREADS=1',
                       shlex.join(command), ''])
    remote('cat > '+shlex.quote(server+'/run_probe.sh'), shell)
    pid = remote('nohup bash '+shlex.quote(server+'/run_probe.sh')+' > '+shlex.quote(server+'/probe.log')
                 +' 2>&1 </dev/null & echo $!').strip()
    assert pid.isdigit()
    state.update(phase='running', launch_pid=pid)
    args.job.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(state))


if __name__ == '__main__':
    main()
