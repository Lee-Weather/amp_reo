"""Optional private runtime teacher input for the exp1.15t comparison."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--teacher_url_b64')
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    from train import get_args, task_registry
    from humanoid.amp.checkpoint import prepare_warm_start
    from humanoid.amp.teacher_transfer import apply_mlp_teacher
    args = get_args()
    if args.task != 'x1_amp_exp115t' or not known.teacher_url_b64:
        raise ValueError('This entrypoint requires the exp1.15t private teacher input')
    # Keep the old MLP file entirely in memory: the SDK must only discover
    # the persistent input/output checkpoint directory, never a temporary PT.
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
    prepare_warm_start(args)
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    apply_mlp_teacher(runner, io.BytesIO(teacher_bytes), train_cfg.runner.teacher_sha256)
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
