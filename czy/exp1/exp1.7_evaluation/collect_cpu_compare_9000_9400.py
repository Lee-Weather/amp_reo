"""Collect the recorded CPU comparison; no launch or training mutations."""
import hashlib
import json
from pathlib import Path
from launch_cpu_compare_9000_9400 import remote, RECORD, ROOT


def main():
    state_path = RECORD/'cpu_compare_9000_9400.json'
    state = json.loads(state_path.read_text())
    source = (ROOT/'scripts/amp_tools/analyze_replay.py').read_text()
    runtime = source.replace(', weights_only=False', '')  # Server torch 1.12 compatibility.
    program = '''
import csv,hashlib,json,os,pathlib
import numpy as np
state=json.loads(__STATE__)
root=pathlib.Path(state['server_dir'])
pid=int(state['launch_pid'])
try:
    os.kill(pid,0)
    alive=pathlib.Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z'
except ProcessLookupError:
    alive=False
complete='CPU_COMPARE_OK' in (root/'cpu_compare.log').read_text()
if alive or not complete:
    print(json.dumps(dict(alive=alive,complete=complete)));raise SystemExit(0)
ns={'__file__':str(root/'scripts/amp_tools/analyze_replay.py')}
exec(__ANALYZER__,ns)
cases=[]
for case in state['checks']:
    model=root/'checkpoints'/('model_%d.pt'%case['iteration'])
    assert hashlib.sha256(model.read_bytes()).hexdigest()==state['checkpoint_sha256'][str(case['iteration'])]
    output=root/case['output'];csvpath=output/'isaac_diag.csv'
    report=ns['evaluate_csv'](csvpath,ns['load_context'](model))
    assert report['rows']==case['steps'] and report['command_observation_max_error']==0
    log=(root/(case['output'].replace('/','_')+'.log')).read_text()
    assert 'Physics Device: cpu' in log and 'GPU Pipeline: disabled' in log
    metadata=json.loads((output/'replay_metadata.json').read_text())
    selected={key:report[key] for key in ['rows','resets','timeouts','non_timeout_terminations','walking','stable_minimum_height']}
    selected.update(case,checkpoint_sha256=state['checkpoint_sha256'][str(case['iteration'])],
                    csv_sha256=hashlib.sha256(csvpath.read_bytes()).hexdigest(),
                    cpu_physics_verified=True,cpu_pipeline_verified=True,
                    replay_metadata=metadata)
    rows=list(csv.DictReader(csvpath.open()))
    late=[r for r in rows if 3.6<=float(r['time_s'])<=19.8]
    vx=np.array([float(r['base_vel_x']) for r in late])
    selected['late_window']={'start_s':float(late[0]['time_s']),'end_s':float(late[-1]['time_s']),
        'vx_p05':float(np.quantile(vx,.05)),'vx_p95':float(np.quantile(vx,.95)),
        'mean_abs_vx':float(np.abs(vx).mean()),
        'vx_below_minus_005_fraction':float((vx<-.05).mean()),
        'vx_above_plus_005_fraction':float((vx>.05).mean()),
        'mean_vy':float(np.mean([float(r['base_vel_y']) for r in late])),
        'mean_wz':float(np.mean([float(r['base_ang_vel_z']) for r in late])),
        'heading_change_deg':float(np.rad2deg(np.diff(np.unwrap([float(r['base_euler_z']) for r in late]))).sum()),
        'note':'Fixed time window, not complete-cycle averaging; includes any reset discontinuities.'}
    cases.append(selected)
result=dict(alive=False,complete=True,scope=state['purpose'],cases=cases,
            source_analyzer_sha256=__SOURCE_SHA__,runtime_analyzer_sha256=__RUNTIME_SHA__,
            compatibility_change='Remove weights_only=False keyword for torch 1.12 only')
print(json.dumps(result))
'''.replace('__STATE__', repr(json.dumps(state))).replace('__ANALYZER__', repr(runtime))
    program = program.replace('__SOURCE_SHA__', repr(hashlib.sha256(source.encode()).hexdigest()))
    program = program.replace('__RUNTIME_SHA__', repr(hashlib.sha256(runtime.encode()).hexdigest()))
    report = json.loads(remote(program, python='/home/robot/Anaconda/envs/F1/bin/python'))
    if not report['complete'] or report['alive']:
        print(json.dumps(report));return
    (RECORD/'cpu_compare_9000_9400_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    state['phase'] = 'completed_and_analyzed'
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps([dict(iteration=c['iteration'],vx=c['vx'],reset_mode=c['reset_mode'],
                           failures=c['non_timeout_terminations'],walking=c['walking'],
                           late_window=c['late_window']) for c in report['cases']]))


if __name__ == '__main__':
    main()
