"""Launch a frozen-policy CPU comparison without sharing the training GPU."""
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
SSH = 'C:/Windows/System32/OpenSSH/ssh.exe'
HOST = 'robot@10.12.201.5'


def remote(program, python='python3'):
    result = subprocess.run([SSH, '-o', 'BatchMode=yes', HOST, python+' -'],
                            input=program.encode(), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-4000:])
    return result.stdout.decode().strip()


def main():
    state_path = RECORD/'cpu_compare_9000_9400.json'
    if state_path.exists():
        raise FileExistsError('Inspect the recorded job; do not launch a duplicate')
    training = json.loads((RECORD.parent/'exp1.7_training.json').read_text())
    receipt = next(r for r in training['status']['saved_checkpoints'] if r['iteration'] == 9400)
    server = '/home/robot/czy/exp1/exp_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    cases = [dict(iteration=n, vx=v, reset_mode=reset, steps=1000, seed=5,
                  output='logs/cpu_{}_{}'.format(n, label))
             for n in (9000, 9400)
             for v, reset, label in ((-.2, 'standing', 'reverse'),
                                      (-.2, 'rsi', 'reverse_rsi'),
                                      (.6, 'standing', 'forward'))]
    state = dict(experiment='exp1.7', purpose='CPU frozen-policy comparison; not GPU acceptance',
                 server_dir=server, training_dir=training['server_dir'],
                 training_pid=training['launch_pid'], code_sha256=training['code_sha256'],
                 checkpoint_sha256={'9000': training['source_checkpoint_sha256'], '9400': receipt['sha256']},
                 sim_device='cpu', pipeline='cpu', rl_device='cpu', num_threads=1,
                 checks=cases, phase='preparing')
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    prepare = '''
import hashlib,json,pathlib,shutil
state=json.loads(__STATE__)
root=pathlib.Path(state['server_dir']); root.mkdir(exist_ok=False)
source=pathlib.Path(state['training_dir'])
shutil.copytree(source/'humanoid',root/'humanoid',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(source/'setup.py',root/'setup.py')
(root/'resources').symlink_to(source/'resources',target_is_directory=True)
h=hashlib.sha256()
for p in sorted(list((root/'humanoid').rglob('*.py'))+[root/'setup.py'],key=lambda p:p.relative_to(root).as_posix()):
    h.update(p.relative_to(root).as_posix().encode());h.update(b'\\0');h.update(p.read_bytes());h.update(b'\\0')
assert h.hexdigest()==state['code_sha256'],h.hexdigest()
(root/'checkpoints').mkdir()
for iteration,src in [(9000,source/'checkpoints/model_9000.pt'),(9400,pathlib.Path(__MIDPOINT__))]:
    dst=root/'checkpoints'/('model_%d.pt'%iteration);shutil.copy2(src,dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest()==state['checkpoint_sha256'][str(iteration)]
lines=['#!/bin/bash','set -euo pipefail','source /home/robot/Anaconda/etc/profile.d/conda.sh','conda activate F1',
       'cd '+str(root),'export PYTHONPATH='+str(root),'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""']
for case in state['checks']:
    command=('timeout 180 python -u humanoid/scripts/play_amp.py --task=x1_amp_exp17 --headless '
             '--num_envs=1 --seed=5 --sim_device=cpu --pipeline=cpu --rl_device=cpu --num_threads=1 '
             '--no_video --checkpoint_path=checkpoints/model_{iteration}.pt --reset_mode={reset_mode} '
             '--steps={steps} --vx={vx} --output_dir={output}').format(**case)
    lines.append(command+' > '+case['output'].replace('/','_')+'.log 2>&1')
lines.append('echo CPU_COMPARE_OK')
(root/'cpu_compare.sh').write_text('\\n'.join(lines)+'\\n')
(root/'comparison_manifest.json').write_text(json.dumps(state,indent=2)+'\\n')
print(json.dumps({'prepared':str(root),'code_sha256':h.hexdigest()}))
'''.replace('__STATE__', repr(json.dumps(state))).replace('__MIDPOINT__', repr(receipt['path']))
    print(remote(prepare))
    launched = subprocess.run([SSH, '-o', 'BatchMode=yes', HOST,
                               'nohup bash '+server+'/cpu_compare.sh > '+server+'/cpu_compare.log 2>&1 </dev/null & echo $!'],
                              capture_output=True, check=True, timeout=30)
    pid = launched.stdout.decode().strip()
    if not pid.isdigit():
        raise RuntimeError('Inspect prepared directory before any further launch: '+repr(pid))
    state.update(phase='running', launch_pid=pid)
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(dict(server_dir=server, launch_pid=pid, cases=len(cases))))


if __name__ == '__main__':
    main()
