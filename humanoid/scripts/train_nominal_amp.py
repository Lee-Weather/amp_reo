"""Private nominal-startup continuation with complete policy/AMP/Adam restoration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.exp212_transfer import restore_nominal_continuation
from humanoid.scripts.resume_amp import main

if __name__=='__main__':main(restore=restore_nominal_continuation)
