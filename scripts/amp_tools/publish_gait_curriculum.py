"""Publish exp1.15 branches using an isolated index and tested immutable snapshot."""
import argparse
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

ROOT = Path(__file__).resolve().parents[2]
PARENT = '91feb9a9102c7b5f35dafbf750eae52a434cc666'
PROXY = 'socks5h://10.12.201.122:39009'


def run(args, *, env=None, stdin=None, cwd=ROOT):
    p = subprocess.run(args, input=stdin, env=env, cwd=cwd, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode('utf-8', 'replace')[-4500:]+p.stdout.decode('utf-8', 'replace')[-1500:])
    return p.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('115', '115t'), required=True)
    parser.add_argument('--push', action='store_true')
    a = parser.parse_args()
    experiment = 'exp1.15'+('t' if a.route.endswith('t') else '')
    branch = 'codex/exp1-15'+('t' if a.route.endswith('t') else '')
    directory = ROOT/'czy/exp1/records'/(experiment+'_evaluation')
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'publication.json'
    if output.exists():
        raise ValueError('Publication already recorded; inspect/resume its branch instead of regenerating')
    original = dict(head=run(['git', 'rev-parse', 'HEAD']).strip().decode(),
                    branch=run(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode(),
                    index_sha256=hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest())
    files = ['humanoid/amp/gait_guidance.py', 'humanoid/amp/teacher_transfer.py', 'humanoid/amp/runner.py',
             'humanoid/envs/x1/x1_amp_exp115_config.py', 'humanoid/envs/x1/x1_amp_exp115t_config.py',
             'humanoid/envs/x1/x1_amp_exp115_env.py', 'humanoid/scripts/train_guided_amp.py',
             'humanoid/envs/__init__.py', 'humanoid/scripts/play_amp.py', 'humanoid/scripts/smoke_amp.py',
             'scripts/amp_tools/preflight_gait_curriculum.py', 'tests/test_exp115.py']
    registry = run(['git', 'show', PARENT+':humanoid/envs/__init__.py']).decode()
    registry += ('\nfrom .x1.x1_amp_exp115_config import X1AMPExp115Cfg, X1AMPExp115CfgPPO\n'
                 'from .x1.x1_amp_exp115t_config import X1AMPExp115tCfg, X1AMPExp115tCfgPPO\n'
                 'from .x1.x1_amp_exp115_env import X1AMPExp115Env\n'
                 'task_registry.register("x1_amp_exp115", X1AMPExp115Env, X1AMPExp115Cfg(), X1AMPExp115CfgPPO())\n'
                 'task_registry.register("x1_amp_exp115t", X1AMPExp115Env, X1AMPExp115tCfg(), X1AMPExp115tCfgPPO())\n')
    overrides = {'humanoid/envs/__init__.py': registry}
    for name in ('play_amp.py', 'smoke_amp.py'):
        path = 'humanoid/scripts/'+name
        contents = run(['git', 'show', PARENT+':'+path]).decode()
        contents = contents.replace('"x1_amp_exp114")', '"x1_amp_exp114", "x1_amp_exp115", "x1_amp_exp115t")')
        if name == 'play_amp.py':
            anchor = '    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)'
            assert anchor in contents
            contents = contents.replace(anchor, "    if args.task in ('x1_amp_exp115', 'x1_amp_exp115t'):\n        cfg.commands.guidance_evaluation = True\n"+anchor)
        overrides[path] = contents
    with tempfile.TemporaryDirectory(prefix='.amp-gait-publication-', dir=ROOT) as temporary:
        temp = Path(temporary).resolve()
        assert temp.is_relative_to(ROOT.resolve())
        env = os.environ.copy()
        env.update(GIT_INDEX_FILE=str(temp/'index'), GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
        run(['git', 'read-tree', PARENT], env=env)
        run(['git', 'add', '--', *files], env=env)
        for path, contents in overrides.items():
            blob = run(['git', 'hash-object', '-w', '--stdin'], stdin=contents.encode()).strip().decode()
            run(['git', 'update-index', '--cacheinfo', '100644,'+blob+','+path], env=env)
        changed = run(['git', 'diff', '--cached', '--name-only', PARENT], env=env).decode().splitlines()
        assert set(changed) == set(files), changed
        run(['git', 'diff', '--cached', '--check', PARENT], env=env)
        tree = run(['git', 'write-tree'], env=env).strip().decode()
        message = 'Add '+experiment+' normalized temporal gait guidance and '+('exact forward teacher initialization' if a.route.endswith('t') else 'stable DH policy transfer')+'\n'
        commit = run(['git', 'commit-tree', tree, '-p', PARENT], stdin=message.encode()).strip().decode()
        archive = io.BytesIO(run(['git', 'archive', '--format=zip', commit, 'humanoid', 'tests', 'scripts', 'resources']))
        checkout = temp/'checkout'; checkout.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert all((checkout/p).resolve().is_relative_to(checkout) for p in z.namelist())
            z.extractall(checkout)
        tests = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=checkout)
        # unittest writes its summary to stderr; retain it without routine output.
        report = temp/'cpu_preflight.json'
        run([sys.executable, 'scripts/amp_tools/preflight_gait_curriculum.py', '--route', a.route,
             '--source', str(ROOT/'czy/data/exp1.14/model_5300.pt'),
             '--teacher', str(ROOT/'czy/data/exp1.9/model_11250.pt'), '--output', str(report)], cwd=checkout)
        preflight = json.loads(report.read_text()); preflight['tested_commit'] = commit
        (directory/'cpu_preflight.json').write_text(json.dumps(preflight, indent=2)+'\n', encoding='utf-8')
        result = dict(experiment=experiment, branch=branch, commit=commit, parent=PARENT, changed_files=changed,
                      exact_isolated_tree_cpu_tests=True, production_checkpoint_preflight=preflight['verified'],
                      published=False, original=original, created_at=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        if a.push:
            run(['git', 'update-ref', 'refs/heads/'+branch, commit, '0'*40])
            git = ['git', '-c', 'http.proxy='+PROXY, '-c', 'http.sslBackend=openssl']
            run(git+['push', 'origin', 'refs/heads/'+branch+':refs/heads/'+branch], env=env)
            remote = run(git+['ls-remote', '--heads', 'origin', branch], env=env).decode().split()[0]
            assert remote == commit
            result.update(published=True, remote_commit=remote, proxy=PROXY)
        assert hashlib.sha256((ROOT/'.git/index').read_bytes()).hexdigest() == original['index_sha256']
        assert run(['git', 'rev-parse', 'HEAD']).strip().decode() == original['head']
        assert run(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode() == original['branch']
        result['original_index_and_head_preserved'] = True
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
