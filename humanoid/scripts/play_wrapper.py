#!/usr/bin/env python3
"""Playback wrapper: download checkpoint from base64-encoded URL arg, then play."""
import os
import sys
import urllib.request
import subprocess
import base64

if len(sys.argv) < 2:
    print("Usage: play_wrapper.py <base64_url> [additional args...]", file=sys.stderr)
    sys.exit(1)

b64_url = sys.argv[1]
url = base64.urlsafe_b64decode(b64_url.encode()).decode()
local_path = "/workspace/amp_reo/model_11950.pt"

print(f"Downloading checkpoint ({len(url)} chars URL)...")
urllib.request.urlretrieve(url, local_path)
size = os.path.getsize(local_path)
print(f"Downloaded {size} bytes OK")

play_args = [
    "python", "/workspace/amp_reo/humanoid/scripts/play_amp.py",
    "--headless",
    "--output_dir", "/workspace/amp_reo/logs/x1_amp_exp33/playback/exp3.3",
    f"--checkpoint_path={local_path}",
] + sys.argv[2:]

print(f"Running: {' '.join(play_args)}")
result = subprocess.run(play_args, cwd="/workspace/amp_reo")
sys.exit(result.returncode)
