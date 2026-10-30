"""Optional private runtime teacher input for the exp1.15t comparison."""
import argparse
import base64
import hashlib
import io
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def mounted_teacher(name):
    """Locate a selected personal-storage input without scanning system trees."""
    supplied = Path(name)
    roots = [Path('/personal'), Path('/workspace/personal'), Path('/workspace/data'),
             Path('/data'), Path('/mnt/data'), Path('/workspace/amp_reo/personal'),
             Path('/workspace/amp_reo/data'), Path('/workspace')]
    for key, value in os.environ.items():
        if ('DATA' in key or 'STORAGE' in key) and ('PATH' in key or 'DIR' in key) and value.startswith('/'):
            roots.append(Path(value))
    deadline = time.monotonic()+60
    while True:
        candidates = [supplied, *[r/supplied for r in roots], *[r/'personal'/supplied for r in roots]]
        for path in candidates:
            if path.is_file():
                return path.read_bytes()
        # Some platform image revisions bind the selected folder under a
        # generated data subdirectory. Bound traversal to three levels and
        # only the known user workspace/data mount roots.
        for root in roots:
            if not root.is_dir(): continue
            for folder, directories, files in os.walk(str(root)):
                relative = Path(folder).relative_to(root)
                directories[:] = [d for d in directories if not d.startswith('.') and d not in ('proc', 'sys', 'dev', 'site-packages', 'resources', 'node_modules')]
                if len(relative.parts) >= 3: directories[:] = []
                if supplied.name in files:
                    return (Path(folder)/supplied.name).read_bytes()
        if time.monotonic() >= deadline:
            raise FileNotFoundError('Selected personal-storage teacher file is not mounted; inspect the same task')
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--teacher_url_b64')
    parser.add_argument('--teacher_path')
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    from train import get_args, task_registry
    from humanoid.amp.checkpoint import prepare_warm_start
    from humanoid.amp.teacher_transfer import apply_mlp_teacher
    args = get_args()
    if args.task != 'x1_amp_exp115t' or bool(known.teacher_url_b64) == bool(known.teacher_path):
        raise ValueError('This entrypoint requires the exp1.15t private teacher input')
    # Keep the old MLP file entirely in memory: the SDK must only discover
    # the persistent input/output checkpoint directory, never a temporary PT.
    if known.teacher_path:
        teacher_bytes = mounted_teacher(known.teacher_path)
    else:
        try:
            url = base64.urlsafe_b64decode(known.teacher_url_b64).decode()
            if not url.startswith('https://'):
                raise ValueError('HTTPS required')
            with urllib.request.urlopen(url, timeout=90) as response:
                teacher_bytes = response.read(32*1024*1024+1)
        except Exception:
            raise RuntimeError('Private teacher download failed; URL withheld') from None
    _, train_cfg = task_registry.get_cfgs(args.task)
    if hashlib.sha256(teacher_bytes).hexdigest() != train_cfg.runner.teacher_sha256:
        raise ValueError('Teacher download hash mismatch')
    if known.teacher_path and Path(known.teacher_path).resolve().parent == Path(args.checkpoint_dir).resolve():
        args.checkpoint_extra_inputs = {str(Path(known.teacher_path).resolve()): train_cfg.runner.teacher_sha256}
    prepare_warm_start(args)
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    apply_mlp_teacher(runner, io.BytesIO(teacher_bytes), train_cfg.runner.teacher_sha256)
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
