"""Send the reviewed working runtime for a no-learning native-reset GPU probe."""
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
    directory=ROOT/'czy/exp1/records/exp2.1_evaluation'
    state_path=directory/'gpu_probe_job.json'
    if state_path.exists():raise ValueError('Inspect existing probe instead of duplicating GPU work')
    query=subprocess.check_output([SSH,'-o','BatchMode=yes',HOST,
        'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader']).decode()
    active,_=gpu_workloads(query);assert not active,active
    code,_=source_files()
    remote='/home/robot/czy/exp1/exp_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_exp21_probe'
    source=ROOT/'czy/data/exp1.17/model_7000.pt'
    state=dict(server_dir=remote,learning_updates=0,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_file_sha256={n:hashlib.sha256(v).hexdigest() for n,v in code.items()})
    with tempfile.TemporaryDirectory(prefix='exp21_probe_') as temp:
        temp=Path(temp);archive=temp/'probe.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for name,raw in code.items():
                item=tarfile.TarInfo(name);item.size=len(raw);t.addfile(item,io.BytesIO(raw))
            t.add(source,arcname='checkpoints/input_model_7000.pt')
        script=temp/'run.sh'
        script.write_text('\n'.join(['#!/bin/bash','set -e','cd '+remote,'tar -xzf probe.tar.gz',
            'cp -a /home/robot/czy/exp1/exp_20261102_084339/resources resources',
            'source /home/robot/Anaconda/etc/profile.d/conda.sh','conda activate F1','export PYTHONPATH='+remote,
            'python -u humanoid/scripts/preflight_exp21_gpu.py --headless --num_envs=32 --seed=5 --task=x1_amp_exp21 --warm_start=checkpoints/input_model_7000.pt',
            'echo GPU_PROBE_COMPLETE'])+'\n',encoding='utf-8',newline='\n')
        subprocess.run([SSH,'-o','BatchMode=yes',HOST,'mkdir -p '+remote],check=True)
        for path in [archive,script]:subprocess.run([SCP,'-o','BatchMode=yes',str(path),HOST+':'+remote+'/'],check=True)
        state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        try:
            subprocess.run([SSH,'-o','BatchMode=yes',HOST,'cd '+remote+' && nohup bash run.sh > probe.log 2>&1 < /dev/null &'],timeout=15,check=True)
        except subprocess.TimeoutExpired:pass
    print(json.dumps({k:state[k] for k in ['server_dir','learning_updates','source_sha256']}),flush=True)


if __name__=='__main__':main()
