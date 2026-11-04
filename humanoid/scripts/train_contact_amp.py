"""Private cloud contact-domain continuation, preserving full DH/AMP state."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.exp211_transfer import restore_contact_continuation
from humanoid.scripts.resume_amp import main

if __name__=='__main__':main(restore=restore_contact_continuation)
