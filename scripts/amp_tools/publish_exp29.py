"""Publish an isolated exp2.9 tree after CPU and no-learning GPU gates."""
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from publish_gait_curriculum import ROOT,PROXY,run

PARENT='68bf6b4c0a38052362768ca2d12e52afb9d1b191'
BRANCH='codex/exp2-9'
FILES=['humanoid/amp/exp29_curriculum.py','humanoid/amp/exp29_phase.py','humanoid/amp/native_reference.py','humanoid/envs/x1/x1_amp_exp20_env.py','humanoid/envs/x1/x1_amp_exp29_config.py','humanoid/envs/x1/x1_amp_exp29_env.py','humanoid/envs/__init__.py','humanoid/scripts/play_amp.py','tests/test_exp29.py','scripts/amp_tools/sim2sim_amp_native.py','scripts/amp_tools/mujoco_diagnostics.py','czy/exp1/exp2.md']


def main():
    directory=ROOT/'czy/exp1/records/exp2.9_evaluation';output=directory/'publication.json'
    directory.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Publication exists; inspect/resume instead of regenerating')
    original=dict(head=run(['git','rev-parse','HEAD']).strip().decode(),
        branch=run(['git','symbolic-ref','--short','HEAD']).strip().decode(),
        index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='.amp-exp29-publication-',dir=ROOT) as temporary:
        temp=Path(temporary).resolve();assert temp.is_relative_to(ROOT.resolve())
        env=os.environ.copy();env.update(GIT_INDEX_FILE=str(temp/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        run(['git','read-tree',PARENT],env=env);run(['git','add','--',*FILES],env=env)
        changed=run(['git','diff','--cached','--name-only',PARENT],env=env).decode().splitlines()
        assert set(changed)==set(FILES),changed
        run(['git','diff','--cached','--check',PARENT],env=env)
        tree=run(['git','write-tree'],env=env).strip().decode()
        commit=run(['git','commit-tree',tree,'-p',PARENT],stdin=b'Continue exp2.9 with command-scaled transition phase and consistent reference derivatives, preserving rewards and controls\n').strip().decode()
        checkout=temp/'checkout';checkout.mkdir()
        archive=io.BytesIO(run(['git','-c','core.autocrlf=false','archive','--format=zip',commit,'humanoid','tests','scripts','resources','setup.py']))
        with zipfile.ZipFile(archive) as z:
            assert all((checkout/n).resolve().is_relative_to(checkout) for n in z.namelist());z.extractall(checkout)
        env['PYTHONPATH']=os.pathsep.join(str(checkout/p) for p in ['.','tests','scripts/amp_tools'])
        env['EXP21_SOURCE_MODEL']=str(ROOT/'czy/data/exp1.17/model_7000.pt')
        env['EXP23_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.2/model_950.pt')
        env['EXP24_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.3/model_200.pt')
        env['EXP25_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.4/model_1100.pt')
        env['EXP26_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.5/model_3100.pt')
        env['EXP27_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.6/model_5100.pt')
        env['EXP28_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.7/model_6300.pt')
        env['EXP29_SOURCE_MODEL']=str(ROOT/'czy/data/exp2.8/model_7500.pt')
        program="""import unittest,json,sys
l=unittest.TestLoader();s=unittest.TestSuite()
for p in ['test_exp20.py','test_exp28.py','test_exp29.py']:s.addTests(l.discover('tests',pattern=p))
r=unittest.TextTestRunner(verbosity=1).run(s)
print(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful())))
sys.exit(0 if r.wasSuccessful() else 1)
"""
        tests=json.loads(run([sys.executable,'-c',program],env=env,cwd=checkout).decode().splitlines()[-1])
        assert tests['passed'],tests
        result=dict(experiment='exp2.9',branch=BRANCH,commit=commit,parent=PARENT,changed_files=changed,
            exact_isolated_tree_tests=tests,exact_isolated_tree_cpu_preflight=True,
            no_separate_gpu_test=True,cloud_startup_validation_required=True,published=False,original=original,
            created_at=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        run(['git','update-ref','refs/heads/'+BRANCH,commit,'0'*40])
        git=['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl']
        run(git+['push','origin','refs/heads/'+BRANCH+':refs/heads/'+BRANCH],env=env)
        remote=run(git+['ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0];assert remote==commit
        assert hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest()==original['index_sha256']
        assert run(['git','rev-parse','HEAD']).strip().decode()==original['head']
        assert run(['git','symbolic-ref','--short','HEAD']).strip().decode()==original['branch']
        result.update(published=True,remote_commit=remote,original_index_and_head_preserved=True)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
