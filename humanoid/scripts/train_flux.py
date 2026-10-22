"""Flux gm-run entry point: install the separate assets, then start normal training."""
import os
from pathlib import Path
import sys

# gm-run accepts a script path while its initial working directory can differ.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from humanoid.amp.assets import ensure_assets


def main():
    ensure_assets(ROOT)
    # Isaac Gym must precede torch, including imports in the normal train module.
    import isaacgym
    from humanoid.scripts.train import train
    from humanoid.utils import get_args
    args = get_args()
    if args.task != "x1_amp":
        raise ValueError("This Flux entry point is configured for --task=x1_amp")
    print("Flux X1 AMP start: num_envs={} max_iterations={}".format(args.num_envs, args.max_iterations), flush=True)
    train(args)


if __name__ == "__main__":
    main()
