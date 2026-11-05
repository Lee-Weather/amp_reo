"""Publish the tested exp1.16 snapshot without changing the active checkout/index."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from publish_gait_curriculum import run, ROOT, PROXY

PARENT = '2efee2451244a06dd88aec1b22a9c2ae7318ade6'
BRANCH = 'codex/exp1-16'


def main():
    directory = ROOT/'czy/exp1/records/exp1.16_evaluation'
    output = directory/'publication.json'
    if output.exists():
        raise ValueError('Publication exists; resume it instead of regenerating')
    original = dict(head=run(['git', 'rev-parse', 'HEAD']).strip().decode(),
                    branch=run(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode(),
                    index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    files = ['humanoid/amp/phase_rewards.py', 'humanoid/amp/reward_transfer.py',
             'humanoid/amp/runner.py', 'humanoid/amp/replay.py', 'humanoid/envs/__init__.py',
             'humanoid/envs/x1/x1_amp_exp116_config.py', 'humanoid/envs/x1/x1_amp_exp116_env.py',
             'humanoid/scripts/play_amp.py', 'humanoid/scripts/smoke_amp.py',
             'scripts/amp_tools/preflight_phase_refinement.py', 'tests/test_exp116.py', 'tests/test_playback.py']
    registry = run(['git', 'show', PARENT+':humanoid/envs/__init__.py']).decode()
    registry += ('\nfrom .x1.x1_amp_exp116_config import X1AMPExp116Cfg, X1AMPExp116CfgPPO\n'
                 'from .x1.x1_amp_exp116_env import X1AMPExp116Env\n'
                 'task_registry.register("x1_amp_exp116", X1AMPExp116Env, X1AMPExp116Cfg(), X1AMPExp116CfgPPO())\n')
    with tempfile.TemporaryDirectory(prefix='.amp-phase-publication-', dir=ROOT) as temp:
        temp = Path(temp).resolve()
        env = os.environ.copy()
        env.update(GIT_INDEX_FILE=str(temp/'index'), GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
        run(['git', 'read-tree', PARENT], env=env)
        run(['git', 'add', '--', *files], env=env)
        blob = run(['git', 'hash-object', '-w', '--stdin'], stdin=registry.encode()).strip().decode()
        run(['git', 'update-index', '--cacheinfo', '100644,'+blob+',humanoid/envs/__init__.py'], env=env)
        changed = run(['git', 'diff', '--cached', '--name-only', PARENT], env=env).decode().splitlines()
        assert set(changed) == set(files)
        run(['git', 'diff', '--cached', '--check', PARENT], env=env)
        tree = run(['git', 'write-tree'], env=env).strip().decode()
        commit = run(['git', 'commit-tree', tree, '-p', PARENT],
                     stdin=b'Refine phase progress feedback while retaining AMP; add 3000 updates for exp1.16\n').strip().decode()
        archive = io.BytesIO(run(['git', 'archive', '--format=zip', commit, 'humanoid', 'tests', 'scripts', 'resources', 'setup.py']))
        checkout = temp/'checkout'; checkout.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert all((checkout/p).resolve().is_relative_to(checkout) for p in z.namelist())
            z.extractall(checkout)
        run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=checkout)
        report = temp/'cpu_preflight.json'
        run([sys.executable, '-X', 'utf8', 'scripts/amp_tools/preflight_phase_refinement.py',
             '--source', str(ROOT/'czy/data/exp1.15t/model_1000.pt'), '--output', str(report)], cwd=checkout)
        preflight = json.loads(report.read_text()); preflight['tested_commit'] = commit
        assert preflight['verified'] and preflight['planned_final_iteration'] == 4000
        (directory/'cpu_preflight.json').write_text(json.dumps(preflight, indent=2)+'\n', encoding='utf-8')
        result = dict(experiment='exp1.16', branch=BRANCH, commit=commit, parent=PARENT,
                      changed_files=changed, exact_isolated_tree_cpu_tests=True,
                      production_checkpoint_preflight=True, published=False, original=original, proxy=PROXY)
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        run(['git', 'update-ref', 'refs/heads/'+BRANCH, commit, '0'*40])
        git = ['git', '-c', 'http.proxy='+PROXY, '-c', 'http.sslBackend=openssl']
        run(git+['push', 'origin', 'refs/heads/'+BRANCH+':refs/heads/'+BRANCH], env=env)
        remote = run(git+['ls-remote', '--heads', 'origin', BRANCH], env=env).decode().split()[0]
        assert remote == commit
        assert hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest() == original['index_sha256']
        assert run(['git', 'rev-parse', 'HEAD']).strip().decode() == original['head']
        assert run(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode() == original['branch']
        result.update(published=True, remote_commit=remote, original_index_and_head_preserved=True)
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
