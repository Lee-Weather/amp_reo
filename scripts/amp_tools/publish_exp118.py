"""Publish the exp1.18 continuation config from the verified exp1.17 commit."""
import hashlib
import json
from pathlib import Path
import os
import tempfile

from publish_gait_curriculum import run, ROOT, PROXY

PARENT = '57cfd3cc04169c3d6004d3a924ba9ed6269659d4'
BRANCH = 'codex/exp1-18'


def main():
    directory = ROOT/'czy/exp1/records/exp1.18_evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'publication.json'
    if output.exists():
        raise ValueError('Publication exists; inspect it instead of regenerating')
    original = dict(head=run(['git', 'rev-parse', 'HEAD']).strip().decode(),
                    branch=run(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode(),
                    index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    files = ['humanoid/envs/__init__.py', 'humanoid/envs/x1/x1_amp_exp118_config.py',
             'humanoid/scripts/play_amp.py']
    registry = run(['git', 'show', PARENT+':humanoid/envs/__init__.py']).decode()
    registry += ('\nfrom .x1.x1_amp_exp118_config import X1AMPExp118Cfg, X1AMPExp118CfgPPO\n'
                 'task_registry.register("x1_amp_exp118", X1AMPExp117Env, X1AMPExp118Cfg(), X1AMPExp118CfgPPO())\n')
    play = run(['git', 'show', PARENT+':humanoid/scripts/play_amp.py']).decode()
    play = play.replace('"x1_amp_exp116", "x1_amp_exp117"',
                        '"x1_amp_exp116", "x1_amp_exp117", "x1_amp_exp118"')
    play = play.replace("('x1_amp_exp115', 'x1_amp_exp115t', 'x1_amp_exp116', 'x1_amp_exp117')",
                        "('x1_amp_exp115', 'x1_amp_exp115t', 'x1_amp_exp116', 'x1_amp_exp117', 'x1_amp_exp118')")
    assert play.count('x1_amp_exp118') == 6
    with tempfile.TemporaryDirectory(prefix='.amp-exp118-publication-', dir=ROOT) as temp:
        temp = Path(temp).resolve()
        env = os.environ.copy()
        env.update(GIT_INDEX_FILE=str(temp/'index'), GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
        run(['git', 'read-tree', PARENT], env=env)
        run(['git', 'add', '--', files[1]], env=env)
        for path, contents in (('humanoid/envs/__init__.py', registry), ('humanoid/scripts/play_amp.py', play)):
            blob = run(['git', 'hash-object', '-w', '--stdin'], stdin=contents.encode()).strip().decode()
            run(['git', 'update-index', '--cacheinfo', '100644,'+blob+','+path], env=env)
        changed = run(['git', 'diff', '--cached', '--name-only', PARENT], env=env).decode().splitlines()
        assert set(changed) == set(files), changed
        run(['git', 'diff', '--cached', '--check', PARENT], env=env)
        tree = run(['git', 'write-tree'], env=env).strip().decode()
        commit = run(['git', 'commit-tree', tree, '-p', PARENT],
                     stdin=b'Add exp1.18 continuation config for 5000 more AMP updates\n').strip().decode()
        result = dict(experiment='exp1.18', branch=BRANCH, commit=commit, parent=PARENT,
                      changed_files=changed, published=False, original=original, proxy=PROXY,
                      checkpoint_transfer='reward_refinement_preserve_policy_amp',
                      source_task='TASK_20261031_027', source_iteration=7000,
                      additional_updates=5000, target_iteration=12000)
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


if __name__ == '__main__':
    main()
