"""Publish the tested yaw-only refinement without touching the working index."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from publish_gait_curriculum import ROOT, PROXY, run

PARENT='9c281a7e5ed43f08eb70353d5aa030066fe0e93a'
BRANCH='codex/exp2-10'
EXPERIMENT='exp2.10'
TEST_PATTERNS=['test_exp210.py','test_exp29_long.py']
MESSAGE=b'Refine exp2.10 yaw stability with full13500 checkpoint continuation; preserve AMP and controls\n'
ENTRIES=['humanoid/scripts/train_yaw_amp.py','humanoid/scripts/resume_amp.py','humanoid/amp/exp210_transfer.py',
         'humanoid/envs/x1/x1_amp_exp210_config.py','humanoid/envs/x1/x1_amp_exp210_env.py','humanoid/envs/__init__.py']
FILES=['humanoid/amp/exp210_transfer.py','humanoid/envs/x1/x1_amp_exp210_config.py',
    'humanoid/envs/x1/x1_amp_exp210_env.py','humanoid/scripts/train_yaw_amp.py',
    'humanoid/scripts/resume_amp.py','tests/test_exp210.py','czy/exp1/exp2.md']
REGISTRATION="""
from .x1.x1_amp_exp210_config import X1AMPExp210Cfg,X1AMPExp210CfgPPO
from .x1.x1_amp_exp210_env import X1AMPExp210Env

task_registry.register('x1_amp_exp210',X1AMPExp210Env,X1AMPExp210Cfg(),X1AMPExp210CfgPPO())
"""
OUTPUT_DIRECTORY=None


def main():
    directory=OUTPUT_DIRECTORY or ROOT/'czy/exp1/records'/(EXPERIMENT+'_evaluation')
    directory.mkdir(parents=True,exist_ok=True)
    output=directory/'publication.json'
    original=dict(head=run(['git','rev-parse','HEAD']).decode().strip(),
        branch=run(['git','symbolic-ref','--short','HEAD']).decode().strip(),
        index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    git=['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl']
    if output.exists():
        result=json.loads(output.read_text())
        assert result['exact_isolated_tree_cpu_preflight']
        commit=result['commit']
    else:
        with tempfile.TemporaryDirectory(prefix='.exp210-publication-',dir=ROOT) as temporary:
            temp=Path(temporary).resolve();assert temp.is_relative_to(ROOT.resolve())
            isolated=dict(env,GIT_INDEX_FILE=str(temp/'index'))
            run(['git','read-tree',PARENT],env=isolated)
            run(['git','add','--',*FILES],env=isolated)
            if REGISTRATION:
                registry=run(['git','show',PARENT+':humanoid/envs/__init__.py']).rstrip()+b'\n'+REGISTRATION.encode()
                blob=run(['git','hash-object','-w','--stdin'],stdin=registry).decode().strip()
                run(['git','update-index','--add','--cacheinfo','100644',blob,'humanoid/envs/__init__.py'],env=isolated)
            changed=run(['git','diff','--cached','--name-only',PARENT],env=isolated).decode().splitlines()
            assert set(changed)==set(FILES+(['humanoid/envs/__init__.py'] if REGISTRATION else [])),changed
            run(['git','diff','--cached','--check',PARENT],env=isolated)
            tree=run(['git','write-tree'],env=isolated).decode().strip()
            commit=run(['git','commit-tree',tree,'-p',PARENT],
                stdin=MESSAGE).decode().strip()
            checkout=temp/'snapshot';checkout.mkdir()
            archive=io.BytesIO(run(['git','-c','core.autocrlf=false','archive','--format=zip',commit,
                                   'humanoid','tests','scripts','resources','setup.py']))
            with zipfile.ZipFile(archive) as z:
                assert all((checkout/n).resolve().is_relative_to(checkout) for n in z.namelist());z.extractall(checkout)
            for name in ENTRIES:
                run(['git','cat-file','-e',commit+':'+name])
                compile((checkout/name).read_bytes(),name,'exec')
            testing=dict(env,PYTHONPATH=os.pathsep.join(str(checkout/p) for p in ['.','tests','scripts/amp_tools']),
                EXP210_SOURCE_MODEL=str(ROOT/'czy/data/exp2.9/model_13500.pt'),
                EXP29_LONG_SOURCE=str(ROOT/'czy/data/exp2.9/model_8500.pt'))
            program="""import unittest,json,sys
l=unittest.TestLoader();s=unittest.TestSuite()
for p in ['test_exp210.py','test_exp29_long.py']:s.addTests(l.discover('tests',pattern=p))
r=unittest.TextTestRunner(verbosity=1).run(s)
print(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful())))
sys.exit(0 if r.wasSuccessful() else 1)
"""
            program=program.replace("['test_exp210.py','test_exp29_long.py']",repr(TEST_PATTERNS))
            tests=json.loads(run([sys.executable,'-c',program],env=testing,cwd=checkout).decode().splitlines()[-1])
            assert tests['passed']
            result=dict(experiment=EXPERIMENT,commit=commit,parent=PARENT,branch=BRANCH,changed_files=changed,
                exact_isolated_tree_tests=tests,exact_isolated_tree_cpu_preflight=True,entry_and_dependencies_verified=True,
                no_gpu_short_training=True,published=False,original=original,created_at=datetime.now(timezone.utc).isoformat())
            output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    existing=run(['git','for-each-ref','--format=%(objectname)','refs/heads/'+BRANCH]).decode().strip()
    if existing:
        assert existing==commit,'Existing branch differs; do not force overwrite'
    else:
        run(['git','update-ref','refs/heads/'+BRANCH,commit,'0'*40])
    run(git+['push','origin','refs/heads/'+BRANCH+':refs/heads/'+BRANCH],env=env)
    remote=run(git+['ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
    assert remote==commit
    assert hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest()==original['index_sha256']
    assert run(['git','rev-parse','HEAD']).decode().strip()==original['head']
    assert run(['git','symbolic-ref','--short','HEAD']).decode().strip()==original['branch']
    result.update(published=True,remote_commit=remote,original_index_and_head_preserved=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(commit=commit,branch=BRANCH,tests=result['exact_isolated_tree_tests'],published=True)),flush=True)


if __name__=='__main__':main()
