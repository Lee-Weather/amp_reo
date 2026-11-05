"""Launch or collect a recorded frozen-policy CPU comparison beside GPU training."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from launch_server_amp import HOST, SSH, remote

ROOT = Path(__file__).resolve().parents[2]


def launch(args):
    training=json.loads(args.state.read_text())
    assert training['host']==HOST and training['backend']=='ssh_server'
    source=training['source_iteration'];candidate=args.iteration
    receipt=next(r for r in training['status']['saved_checkpoints'] if r['iteration']==candidate)
    assert source<candidate<=training['target_iteration']
    assert receipt['path'].startswith(training['server_dir']+'/')
    assert str(candidate) in training['verified_checkpoints']
    args.record_dir.mkdir(parents=True,exist_ok=True)
    job_path=args.record_dir/('cpu_compare_%d_%d.json'%(source,candidate))
    if job_path.exists():raise FileExistsError('Inspect recorded process; do not launch a duplicate')
    server='/home/robot/czy/exp1/exp_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    checks=[]
    for iteration in (source,candidate):
        for label,mode,vx,steps in [('reverse','standing',-.2,1000),('reverse_rsi','rsi',-.2,1000),('profile','standing',None,3000)]:
            checks.append(dict(iteration=iteration,reset_mode=mode,vx=vx,steps=steps,seed=5,
                               output='logs/cpu_%d_%s'%(iteration,label)))
    state=dict(experiment=training['experiment'],purpose='Frozen policy comparison with identical CPU physics/config, not GPU acceptance',
               server_dir=server,training_dir=training['server_dir'],training_pid=training['launch_pid'],
               code_sha256=training['code_sha256'],task=training['task'],checks=checks,phase='preparing',
               checkpoint_sha256={str(source):training['source_checkpoint_sha256'],str(candidate):receipt['sha256']},
               source_iteration=source,candidate_iteration=candidate,sim_device='cpu',pipeline='cpu',rl_device='cpu',num_threads=1)
    job_path.write_text(json.dumps(state,indent=2)+'\n')
    program='''
import pathlib,shutil,hashlib,json
s=json.loads(__STATE__);root=pathlib.Path(s['server_dir']);root.mkdir(exist_ok=False)
training=pathlib.Path(s['training_dir'])
shutil.copytree(training/'humanoid',root/'humanoid',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(training/'setup.py',root/'setup.py');(root/'resources').symlink_to(training/'resources',target_is_directory=True)
h=hashlib.sha256()
for p in sorted((root/'humanoid').rglob('*.py'))+[root/'setup.py']:
    h.update(p.relative_to(root).as_posix().encode()+b'\\0'+p.read_bytes()+b'\\0')
assert h.hexdigest()==s['code_sha256']
(root/'checkpoints').mkdir()
for iteration,path in __MODELS__:
    dst=root/'checkpoints'/('model_%d.pt'%iteration);shutil.copy2(path,dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest()==s['checkpoint_sha256'][str(iteration)]
print('CPU_COMPARISON_PREPARED')
'''.replace('__STATE__',repr(json.dumps(state))).replace('__MODELS__',repr([
        (source,training['server_dir']+'/checkpoints/model_%d.pt'%source),(candidate,receipt['path'])]))
    remote('python3 -',program)
    lines=['set -euo pipefail','cd '+shlex.quote(server),
           'source /home/robot/Anaconda/etc/profile.d/conda.sh','conda activate F1',
           'export PYTHONPATH='+shlex.quote(server),'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""']
    for case in checks:
        command=['timeout','300','python','-u','humanoid/scripts/play_amp.py','--task='+state['task'],
                 '--headless','--num_envs=1','--seed=5','--sim_device=cpu','--pipeline=cpu','--rl_device=cpu',
                 '--num_threads=1','--no_video','--checkpoint_path=checkpoints/model_%d.pt'%case['iteration'],
                 '--reset_mode='+case['reset_mode'],'--steps='+str(case['steps']),'--output_dir='+case['output']]
        if case['vx'] is not None:command.append('--vx='+str(case['vx']))
        lines.append(shlex.join(command)+' > '+case['output'].replace('/','_')+'.log 2>&1')
    lines+=['echo CPU_COMPARE_OK','']
    remote('cat > '+shlex.quote(server+'/cpu_compare.sh'),'\n'.join(lines))
    pid=remote('nohup bash '+shlex.quote(server+'/cpu_compare.sh')+' > '+shlex.quote(server+'/cpu_compare.log')+' 2>&1 </dev/null & echo $!').strip()
    assert pid.isdigit()
    state.update(phase='running',launch_pid=pid);job_path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(job=str(job_path),server_dir=server,launch_pid=pid,cases=len(checks))))


def collect(args):
    state=json.loads(args.job.read_text());assert state['server_dir'].startswith('/home/robot/czy/exp1/exp_')
    analyzer=(ROOT/'scripts/amp_tools/analyze_replay.py').read_text()
    runtime_analyzer=analyzer.replace(', weights_only=False','')
    acceptance=(ROOT/'scripts/amp_tools/summarize_amp_acceptance.py').read_text()
    node=next(n for n in ast.parse(acceptance).body if isinstance(n,ast.FunctionDef) and n.name=='braking_report')
    braking=ast.get_source_segment(acceptance,node)
    program='''
import pathlib,json,hashlib,math,csv
import numpy as np
s=json.loads(__STATE__);root=pathlib.Path(s['server_dir']);proc=pathlib.Path('/proc/%s/stat'%s['launch_pid'])
alive=proc.exists() and proc.read_text().split()[2]!='Z'
complete='CPU_COMPARE_OK' in (root/'cpu_compare.log').read_text()
if alive or not complete:
    print(json.dumps(dict(alive=alive,complete=complete)));raise SystemExit(0)
ns={'__file__':str(root/'scripts/amp_tools/analyze_replay.py')};exec(__ANALYZER__,ns)
bn={'np':np,'math':math};exec(__BRAKING__,bn)
contexts={};cases=[]
for case in s['checks']:
    iteration=case['iteration'];model=root/'checkpoints'/('model_%d.pt'%iteration)
    assert hashlib.sha256(model.read_bytes()).hexdigest()==s['checkpoint_sha256'][str(iteration)]
    if iteration not in contexts:contexts[iteration]=ns['load_context'](model)
    context=contexts[iteration];path=root/case['output']/'isaac_diag.csv'
    metadata=json.loads((path.parent/'replay_metadata.json').read_text())
    assert metadata['checkpoint']==iteration and metadata['captured_steps']==case['steps']
    assert metadata['seed']==5 and not metadata['noise'] and not metadata['pushes'] and metadata['reset_mode']==case['reset_mode']
    assert metadata['task']==s['task'] and metadata['video_frames']==0
    expected=[(case['steps'],case['vx'])] if case['vx'] is not None else [(500,v) for v in (0.,.4,.6,0.,.4,0.)]
    assert len(metadata['profile'])==len(expected)
    assert all(actual[0]==steps and abs(actual[1]-vx)<1e-6 for actual,(steps,vx) in zip(metadata['profile'],expected))
    assert metadata['policy_metadata']==context['metadata']
    log=(root/(case['output'].replace('/','_')+'.log')).read_text()
    assert 'Physics Device: cpu' in log and 'GPU Pipeline: disabled' in log and 'Replay files closed: 0 frames' in log
    report=ns['evaluate_csv'](path,context)
    assert report['rows']==case['steps'] and report['command_observation_max_error']==0
    selected={k:report[k] for k in ('rows','non_timeout_terminations','failures','standing','walking','speed_04','speed_06','raw_target_clip_rate')}
    data=np.genfromtxt(path,delimiter=',',names=True)
    selected.update(case,braking=bn['braking_report'](data,context['metadata']['policy_dt']),
                    csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu_physics_verified=True,cpu_pipeline_verified=True)
    cases.append(selected)
print(json.dumps(dict(alive=False,complete=True,cases=cases)))
'''.replace('__STATE__',repr(json.dumps(state))).replace('__ANALYZER__',repr(runtime_analyzer)).replace('__BRAKING__',repr(braking))
    output=remote('/home/robot/Anaconda/envs/F1/bin/python -',program)
    result=json.loads(output)
    if result['complete'] and not result['alive']:
        result.update(scope=state['purpose'],checkpoint_sha256=state['checkpoint_sha256'],code_sha256=state['code_sha256'],
                      source_analyzer_sha256=hashlib.sha256(analyzer.encode()).hexdigest(),
                      runtime_analyzer_sha256=hashlib.sha256(runtime_analyzer.encode()).hexdigest(),
                      braking_function_sha256=hashlib.sha256(braking.encode()).hexdigest(),
                      limits=['CPU physics differs from GPU; compare only the two models under the same CPU setup.',
                              'Frozen discriminator differs between saved models; do not compare their total reward as one fixed objective.',
                              'This diagnostic does not replace final full-range GPU acceptance.'])
        args.job.with_name(args.job.stem+'_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        state['phase']='completed_and_analyzed';args.job.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps([dict(iteration=c['iteration'],vx=c['vx'],mode=c['reset_mode'],falls=c['non_timeout_terminations'],walking=c['walking'],braking=c['braking']) for c in result['cases']]))
    else:print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    start=sub.add_parser('launch');start.add_argument('--state',type=Path,required=True)
    start.add_argument('--iteration',type=int,required=True);start.add_argument('--record-dir',type=Path,required=True)
    end=sub.add_parser('collect');end.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();launch(args) if args.mode=='launch' else collect(args)


if __name__=='__main__':main()
