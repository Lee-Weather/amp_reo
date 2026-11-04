"""Cloud exp2.10 entry: verified13500 full-state yaw-only refinement."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.exp210_transfer import restore_yaw_continuation
from humanoid.scripts.resume_amp import main


if __name__ == '__main__':
    main(restore=restore_yaw_continuation)
