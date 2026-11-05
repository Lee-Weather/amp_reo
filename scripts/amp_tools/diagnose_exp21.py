"""Deploy three immutable, no-learning baselines for exp2.1."""
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from replay_archive import ROOT,SSH,SCP,HOST,source_files,gpu_workloads


def main():
    directory=ROOT/'czy/exp1/records/exp2.1_evaluation';directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'diagnostic_job.json'
    if state_path.exists(): raise ValueError('Inspect existing diagnostics; do not launch duplicates')
    output=subprocess.check_output([SSH,'-o','BatchMode=yes',HOST,
        'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader']).decode()
    active,_=gpu_workloads(output);assert not active,active
    commit='7e96e4e51d1a2e12e309b24bb8288a718b659511'
    code,_=source_files(commit)
    resources='/home/robot/czy/exp1/exp_20261102_084339/resources'
    remote='/home/robot/czy/exp1/exp_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_exp21_diagnostics'
    cases=[dict(name='source_control',task='x1_amp_exp117',model='model_7000.pt',vx=.3,extra=''),
           dict(name='source_native',task='x1_amp_exp20',model='model_7000.pt',vx=.5953696966,extra=' --source_policy'),
           dict(name='stopped_native',task='x1_amp_exp20',model='model_2100.pt',vx=.5953696966,extra='')]
    state=dict(server_dir=remote,commit=commit,cases=cases,learning_updates=0,
        model_sha256={name:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for name,path in
        [('model_7000.pt','czy/data/exp1.17/model_7000.pt'),('model_2100.pt','czy/data/exp2.0/model_2100.pt')]})
    with tempfile.TemporaryDirectory(prefix='exp21_diagnostic_') as temp:
        temp=Path(temp);archive=temp/'diagnostic.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for name,raw in code.items():
                item=tarfile.TarInfo(name);item.size=len(raw);t.addfile(item,io.BytesIO(raw))
            for experiment,iteration in [('exp1.17',7000),('exp2.0',2100)]:
                t.add(ROOT/'czy/data'/experiment/('model_'+str(iteration)+'.pt'),arcname='checkpoints/model_'+str(iteration)+'.pt')
        script=temp/'run.sh'
        lines=['#!/bin/bash','set -e','cd '+remote,'tar -xzf diagnostic.tar.gz',
               'cp -a '+resources+' resources','source /home/robot/Anaconda/etc/profile.d/conda.sh',
               'conda activate F1','export PYTHONPATH='+remote]
        for c in cases:
            lines.append('xvfb-run -a python -u humanoid/scripts/play_amp.py --headless --num_envs=1 --seed=5'+
                ' --task='+c['task']+' --checkpoint_path=checkpoints/'+c['model']+' --steps=3000 --vx='+str(c['vx'])+
                ' --output_dir=logs/'+c['name']+c['extra']+' > '+c['name']+'.log 2>&1')
        lines.append("echo DIAGNOSTICS_COMPLETE")
        script.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        subprocess.run([SSH,'-o','BatchMode=yes',HOST,'mkdir -p '+remote],check=True)
        for path in [archive,script]: subprocess.run([SCP,'-o','BatchMode=yes',str(path),HOST+':'+remote+'/'],check=True)
        state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        subprocess.run([SSH,'-o','BatchMode=yes',HOST,'cd '+remote+' && nohup bash run.sh > suite.log 2>&1 < /dev/null &'],check=True,timeout=30)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
