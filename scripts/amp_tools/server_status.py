"""Read the recorded SSH training process, checkpoint receipts and real progress."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["backend"] != "ssh_server" or state["host"] != "robot@10.12.201.5":
        raise ValueError("Use the recorded 201.5 server training state")
    remote = state["server_dir"]
    if not remote.startswith("/home/robot/czy/exp1/exp_"):
        raise ValueError("Unexpected server workspace")
    pid = int(state["launch_pid"])
    program = r'''
import json, pathlib, re
root = pathlib.Path(REMOTE)
log = (root/LOGFILE).read_text(errors='replace')
proc = pathlib.Path('/proc/PID/stat')
alive = proc.exists() and proc.read_text().split()[2] != 'Z'
iterations = re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', log)
receipts = [json.loads(line.split('CHECKPOINT_SAVED ',1)[1]) for line in log.splitlines() if line.startswith('CHECKPOINT_SAVED ')]
restored = [json.loads(line.split('CHECKPOINT_RESTORED ',1)[1]) for line in log.splitlines() if line.startswith('CHECKPOINT_RESTORED ')]
eta = re.findall(r'ETA:\s*([0-9.]+)s', log)
times = re.findall(r'Total timesteps:\s*(\d+)', log)
complete = SUCCESS in log and any(r['iteration']==TARGET for r in receipts)
phase = 'running' if alive else ('completed_training' if complete else 'stopped_requires_inspection')
print(json.dumps(dict(phase=phase, process_alive=alive,
 iteration=int(iterations[-1][0]) if iterations else None,
 target_iteration=TARGET, eta_seconds=float(eta[-1]) if eta else None,
 new_timesteps=int(times[-1]) if times else 0,
 restored_checkpoint=restored[-1] if restored else None,
 saved_checkpoints=receipts, log_tail=log.splitlines()[-12:])))
'''.replace("REMOTE", repr(remote)).replace("LOGFILE", repr(state.get("log_file", "train_recovery.log"))).replace("SUCCESS", repr(state.get("success_marker", "SERVER_RECOVERY_OK"))).replace("PID", str(pid)).replace("TARGET", str(int(state["target_iteration"])))
    result = subprocess.run(["C:/Windows/System32/OpenSSH/ssh.exe", "-o", "BatchMode=yes",
                             state["host"], "python3 -"], input=program.encode(), capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError("Server observation failed; inspect the same handle without restarting: "
                           +result.stderr.decode("utf-8", errors="replace")[-1000:])
    observation = json.loads(result.stdout.decode("utf-8"))
    observation["observed_at"] = datetime.now().astimezone().isoformat()
    state["phase"] = observation["phase"]
    state["status"] = observation
    args.state.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(observation, ensure_ascii=False))


if __name__ == "__main__":
    main()
