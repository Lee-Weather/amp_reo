"""Publish an isolated exp2.1 tree after CPU and no-learning GPU gates."""
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

PARENT='7e96e4e51d1a2e12e309b24bb8288a718b659511'
BRANCH='codex/exp2-1'
FILES=['humanoid/amp/gait_curriculum.py','humanoid/amp/conservative_runner.py','humanoid/amp/runner.py',
       'humanoid/envs/x1/x1_amp_exp21_config.py','humanoid/envs/x1/x1_amp_exp21_env.py',
       'humanoid/envs/__init__.py','humanoid/utils/task_registry.py','humanoid/scripts/play_amp.py',
       'humanoid/scripts/smoke_amp.py','humanoid/scripts/preflight_exp21_gpu.py',
       'tests/test_exp21.py','scripts/amp_tools/preflight_exp21.py','czy/exp1/exp2.md']


def main():
    directory=ROOT/'czy/exp1/records/exp2.1_evaluation';output=directory/'publication.json'
    if output.exists():raise ValueError('Publication exists; inspect/resume instead of regenerating')
    verified=json.loads((directory/'verification.json').read_text());assert verified['passed']
    gpu=json.loads((directory/'gpu_preflight.json').read_text());assert gpu['reset_feasibility_passed']
    probe=json.loads((directory/'gpu_probe_job.json').read_text())
    for path,digest in probe['source_file_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    original=dict(head=run(['git','rev-parse','HEAD']).strip().decode(),
        branch=run(['git','symbolic-ref','--short','HEAD']).strip().decode(),
        index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='.amp-exp21-publication-',dir=ROOT) as temporary:
        temp=Path(temporary).resolve();assert temp.is_relative_to(ROOT.resolve())
        env=os.environ.copy();env.update(GIT_INDEX_FILE=str(temp/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        run(['git','read-tree',PARENT],env=env);run(['git','add','--',*FILES],env=env)
        changed=run(['git','diff','--cached','--name-only',PARENT],env=env).decode().splitlines()
        assert set(changed)==set(FILES),changed
        run(['git','diff','--cached','--check',PARENT],env=env)
        tree=run(['git','write-tree'],env=env).strip().decode()
        commit=run(['git','commit-tree',tree,'-p',PARENT],stdin=b'Add exp2.1 conservative native gait validation with performance gates\n').strip().decode()
        checkout=temp/'checkout';checkout.mkdir()
        archive=io.BytesIO(run(['git','archive','--format=zip',commit,'humanoid','tests','scripts','resources','setup.py']))
        with zipfile.ZipFile(archive) as z:
            assert all((checkout/n).resolve().is_relative_to(checkout) for n in z.namelist());z.extractall(checkout)
        env['PYTHONPATH']=os.pathsep.join(str(checkout/p) for p in ['.','tests','scripts/amp_tools'])
        env['EXP21_SOURCE_MODEL']=str(ROOT/'czy/data/exp1.17/model_7000.pt')
        program="""import unittest,json,sys
l=unittest.TestLoader();s=unittest.TestSuite()
for p in ['test_exp11*.py','test_exp20.py','test_exp21.py','test_playback.py','test_replay_revision.py','test_native_analysis.py']:s.addTests(l.discover('tests',pattern=p))
r=unittest.TextTestRunner(verbosity=1).run(s)
print(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful())))
sys.exit(0 if r.wasSuccessful() else 1)
"""
        tests=json.loads(run([sys.executable,'-c',program],env=env,cwd=checkout).decode().splitlines()[-1])
        assert tests['tests_run']==verified['tests_run'] and tests['passed'],tests
        report=temp/'preflight.json'
        run([sys.executable,'scripts/amp_tools/preflight_exp21.py','--source',env['EXP21_SOURCE_MODEL'],'--output',str(report)],env=env,cwd=checkout)
        assert json.loads(report.read_text())['software_passed']
        result=dict(experiment='exp2.1',branch=BRANCH,commit=commit,parent=PARENT,changed_files=changed,
            exact_isolated_tree_tests=tests,exact_isolated_tree_cpu_preflight=True,
            gpu_reset_probe_passed=True,gpu_probe_code_matches=True,published=False,original=original,
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
