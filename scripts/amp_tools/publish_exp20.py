"""Publish the verified exp2.0 runtime with an isolated Git index."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from publish_gait_curriculum import ROOT, PROXY, run

PARENT = '9e56aaf762d27d33b7c5bd19abdc4cc3ef33477a'
BRANCH = 'codex/exp2-0'


def main():
    directory = ROOT/'czy/exp1/records/exp2.0_evaluation'
    output = directory/'publication.json'
    if output.exists():
        raise ValueError('Publication exists; inspect/resume it instead of creating another commit')
    verified = json.loads((directory/'verification.json').read_text())
    assert verified['passed'] and verified['tests_run'] == 79
    files = list(verified['source_files_sha256']) + list(verified['runtime_data_sha256'])
    # The working registry includes the historical comparison task. Include its
    # missing dependencies so the immutable cloud checkout imports identically.
    files += ['humanoid/amp/reference_distance.py',
              'humanoid/envs/x1/x1_amp_exp114t_config.py',
              'humanoid/envs/x1/x1_amp_exp114t_env.py',
              'tests/test_exp114t.py',
              'scripts/amp_tools/README.md', 'czy/exp1/exp2.md']
    for path, digest in {**verified['source_files_sha256'], **verified['runtime_data_sha256']}.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    original = dict(head=run(['git','rev-parse','HEAD']).strip().decode(),
                    branch=run(['git','symbolic-ref','--short','HEAD']).strip().decode(),
                    index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='.amp-exp20-publication-',dir=ROOT) as temporary:
        temp = Path(temporary).resolve()
        assert temp.is_relative_to(ROOT.resolve())
        env = os.environ.copy()
        env.update(GIT_INDEX_FILE=str(temp/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        run(['git','read-tree',PARENT],env=env)
        run(['git','add','--',*files],env=env)
        changed = run(['git','diff','--cached','--name-only',PARENT],env=env).decode().splitlines()
        assert set(changed).issubset(files) and set(verified['runtime_data_sha256']).issubset(changed)
        run(['git','diff','--cached','--check',PARENT],env=env)
        tree = run(['git','write-tree'],env=env).strip().decode()
        commit = run(['git','commit-tree',tree,'-p',PARENT],
                     stdin=b'Add exp2.0 native full-body AMP curriculum and policy-only transfer\n').strip().decode()
        archive = io.BytesIO(run(['git','archive','--format=zip',commit,'humanoid','tests','scripts','resources','setup.py']))
        checkout = temp/'checkout'; checkout.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert all((checkout/p).resolve().is_relative_to(checkout) for p in z.namelist())
            z.extractall(checkout)
        # Validate exactly the code/resources that the cloud will clone.
        program = """import unittest,json,sys
loader=unittest.TestLoader(); suite=unittest.TestSuite()
for pattern in ['test_exp11*.py','test_exp20.py','test_playback.py','test_replay_revision.py','test_native_analysis.py']:
 suite.addTests(loader.discover('tests',pattern=pattern))
r=unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),passed=r.wasSuccessful())))
sys.exit(0 if r.wasSuccessful() else 1)
"""
        tests = json.loads(run([sys.executable,'-c',program],cwd=checkout).decode().splitlines()[-1])
        assert tests['tests_run']==79 and tests['passed'], tests
        report = temp/'preflight_summary.json'
        run([sys.executable,'scripts/amp_tools/preflight_exp20.py','--source',
             str(ROOT/'czy/data/exp1.17/model_7000.pt'),'--output',str(report)],cwd=checkout)
        preflight = json.loads(report.read_text())
        assert preflight['ready_for_cloud'] and preflight['software_checks_passed']
        result = dict(experiment='exp2.0',branch=BRANCH,commit=commit,parent=PARENT,
                      changed_files=changed,exact_isolated_tree_tests=tests,
                      exact_isolated_tree_cpu_preflight=True,runtime_json_included=True,
                      original=original,published=False,created_at=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        run(['git','update-ref','refs/heads/'+BRANCH,commit,'0'*40])
        git = ['git','-c','http.proxy='+PROXY,'-c','http.sslBackend=openssl']
        run(git+['push','origin','refs/heads/'+BRANCH+':refs/heads/'+BRANCH],env=env)
        remote = run(git+['ls-remote','--heads','origin',BRANCH],env=env).decode().split()[0]
        assert remote==commit
        assert hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest()==original['index_sha256']
        assert run(['git','rev-parse','HEAD']).strip().decode()==original['head']
        assert run(['git','symbolic-ref','--short','HEAD']).strip().decode()==original['branch']
        result.update(published=True,remote_commit=remote,original_index_and_head_preserved=True,proxy=PROXY)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()
