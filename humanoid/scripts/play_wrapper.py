#!/usr/bin/env python3
"""Playback wrapper: download checkpoint from URL, then run play_amp.py."""
import os
import sys
import urllib.request
import subprocess
import base64

# The checkpoint URL is passed as base64-encoded env var
b64_url = os.environ.get("CHECKPOINT_URL_B64")
if not b64_url:
    print("ERROR: CHECKPOINT_URL_B64 env var not set", file=sys.stderr)
    sys.exit(1)

url = base64.urlsafe_b64decode(b64_url.encode()).decode()
local_path = "/workspace/amp_reo/model_11950.pt"

print(f"Downloading checkpoint from URL to {local_path}...")
urllib.request.urlretrieve(url, local_path)
size = os.path.getsize(local_path)
print(f"Downloaded {size} bytes OK")

# Extract task arg from remaining args
task = "x1_amp_exp33"
for a in sys.argv[1:]:
    if a.startswith("--task="):
        task = a.split("=", 1)[1]

# Run play_amp
play_args = [
    "python", "/workspace/amp_reo/humanoid/scripts/play_amp.py",
    f"--task={task}",
    "--headless",
    "--no_video",
    f"--checkpoint_path={local_path}",
]

print(f"Running: {' '.join(play_args)}")
result = subprocess.run(play_args, cwd="/workspace/amp_reo")
sys.exit(result.returncode)
