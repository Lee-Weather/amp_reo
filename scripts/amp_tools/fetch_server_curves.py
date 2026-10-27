"""Read TensorBoard scalars from the exact recorded SSH AMP training workspace."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-observations", action="store_true",
                        help="Also export per-observation mean/std; full events always remain on server")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["backend"] != "ssh_server" or state["host"] != "robot@10.12.201.5":
        raise ValueError("Use the recorded 201.5 training state")
    root = state["server_dir"]
    if not root.startswith("/home/robot/czy/exp1/exp_"):
        raise ValueError("Unexpected server workspace")
    export_path = root+"/curves_export_"+uuid.uuid4().hex+".json"
    program = '''import json,pathlib,hashlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=pathlib.Path(__ROOT__)/'logs'/__TASK__/'exported_data'
runs=sorted({p.parent for p in root.glob('*/events.out.tfevents.*')})
if len(runs)!=1:
 raise ValueError('Expected one actual training event directory, got '+str(len(runs)))
events=EventAccumulator(str(runs[0]),size_guidance={'scalars':0});events.Reload()
tags=events.Tags()['scalars'];selected=[key for key in tags if __INCLUDE_OBS__ or not key.startswith('Observation/')]
series={key:{'steps':[e.step for e in events.Scalars(key)],'values':[e.value for e in events.Scalars(key)]} for key in selected}
path=pathlib.Path(__EXPORT__);path.write_text(json.dumps(dict(event_directory=str(runs[0]),series=series,excluded_scalar_tags=len(tags)-len(selected))))
print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
'''.replace("__ROOT__", repr(root)).replace("__TASK__", repr(state["task"])).replace("__EXPORT__", repr(export_path)).replace("__INCLUDE_OBS__", repr(args.include_observations))
    result = subprocess.run(["C:/Windows/System32/OpenSSH/ssh.exe", "-o", "BatchMode=yes",
                             state["host"], "source /home/robot/Anaconda/etc/profile.d/conda.sh && conda activate F1 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -"],
                            input=program.encode(), capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError("Observation failed; do not restart training: "+result.stderr.decode("utf-8", errors="replace")[-1500:])
    manifest = json.loads(result.stdout.decode("utf-8"))
    if manifest["path"] != export_path:
        raise ValueError("Unexpected export path")
    with tempfile.TemporaryDirectory(prefix="amp_server_curves_") as temp:
        staged = Path(temp)/"curves.json"
        subprocess.run(["C:/Windows/System32/OpenSSH/scp.exe", "-o", "BatchMode=yes",
                        state["host"]+":"+export_path, str(staged)], check=True, timeout=45)
        payload = staged.read_bytes()
        if len(payload) != manifest["bytes"] or hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError("Transferred export does not match source receipt")
        curves = json.loads(payload)
    cleanup = "from pathlib import Path\np=Path("+repr(export_path)+")\nassert p.parent==Path("+repr(root)+") and p.name.startswith('curves_export_')\np.unlink()\n"
    subprocess.run(["C:/Windows/System32/OpenSSH/ssh.exe", "-o", "BatchMode=yes", state["host"], "python3 -"],
                   input=cleanup.encode(), check=True, timeout=20)
    curves["collection"] = "Remote TensorBoard export, SHA256-verified SCP snapshot"
    curves.update(backend="ssh_server", source_iteration=state["source_iteration"],
                  target_iteration=state["target_iteration"], experiment=state["experiment"],
                  observed_at=datetime.now().astimezone().isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(curves, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key:dict(samples=len(value["steps"]), last_step=value["steps"][-1], last=value["values"][-1])
                      for key,value in curves["series"].items()
                      if value["steps"] and (key.startswith("Commands/") or key.startswith("Tracking/")
                                              or key.startswith("Train/") or key in ("AMP/kl", "AMP/kl_max"))}))


if __name__ == "__main__":
    main()
