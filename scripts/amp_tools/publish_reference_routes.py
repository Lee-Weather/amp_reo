"""Publish one independently preflighted reference-guidance branch, preserving the worktree/index."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PARENT = "6869e5f17d9b2ee655db1ff306071ee2cf1020e1"
PROXY = "socks5h://10.12.201.122:39009"


def run(args, *, env=None, stdin=None, cwd=ROOT):
    result = subprocess.run(args, input=stdin, env=env, cwd=cwd, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-3000:]+result.stdout.decode("utf-8", "replace")[-3000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("114", "114t"), required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    route = args.route
    experiment = "exp1.14"+("t" if route.endswith("t") else "")
    branch = "codex/exp1-14"+("t" if route.endswith("t") else "")
    directory = ROOT/"czy/exp1/records"/(experiment+"_evaluation")
    output = directory/"publication.json"
    if output.exists():
        raise ValueError("Publication record already exists; inspect the existing branch instead of republishing")
    original_index = hashlib.sha256((ROOT/".git/index").read_bytes()).hexdigest()
    original_head = run(["git", "rev-parse", "HEAD"]).strip().decode()
    original_branch = run(["git", "symbolic-ref", "--short", "HEAD"]).strip().decode()
    common = ["humanoid/envs/x1/x1_amp_exp114_common_config.py", "humanoid/envs/x1/x1_amp_exp114_common_env.py",
              "humanoid/envs/x1/x1_amp_exp"+route+"_config.py", "humanoid/envs/x1/x1_amp_exp"+route+"_env.py",
              "humanoid/envs/__init__.py", "humanoid/scripts/play_amp.py", "humanoid/scripts/smoke_amp.py",
              "tests/reference_task_fixtures.py", "tests/test_exp"+route+".py",
              "scripts/amp_tools/preflight_reference_transfer.py"]
    files = common+(["humanoid/amp/phase_reference.py", "humanoid/amp/dh_models.py", "humanoid/amp/runner.py",
                     "tests/test_phase_reference.py"] if route == "114" else
                    ["humanoid/amp/reference_distance.py", "tests/test_reference_distance.py"])
    overrides = {}
    parent_registry = run(["git", "show", PARENT+":humanoid/envs/__init__.py"]).decode()
    class_name = "X1AMPExp"+route+"Cfg"
    env_name = "X1AMPExp"+route+"Env"
    overrides["humanoid/envs/__init__.py"] = (parent_registry+"\nfrom .x1.x1_amp_exp"+route+"_config import "+class_name+", "+class_name+"PPO\n"
        +"from .x1.x1_amp_exp"+route+"_env import "+env_name+"\n\ntask_registry.register(\"x1_amp_exp"+route+"\", "+env_name+", "+class_name+"(), "+class_name+"PPO())\n")
    other = "114t" if route == "114" else "114"
    for path in ("humanoid/scripts/play_amp.py", "humanoid/scripts/smoke_amp.py"):
        overrides[path] = (ROOT/path).read_text(encoding="utf-8").replace(', "x1_amp_exp'+other+'"', '')
    if route == "114t":
        overrides["humanoid/scripts/smoke_amp.py"] = overrides["humanoid/scripts/smoke_amp.py"].replace(
            'in ("policy_only", "policy_append_phase"):', '== "policy_only":').replace(
            '            if train_cfg.runner.warm_start_mode == "policy_append_phase":\n'
            '                from humanoid.amp.dh_models import append_phase_policy_state\n'
            '                source_state = append_phase_policy_state(source_state, runner.actor_critic.state_dict())\n', '')
    with tempfile.TemporaryDirectory(prefix=".amp-publication-", dir=ROOT) as tmp:
        temp = Path(tmp).resolve()
        assert temp.is_relative_to(ROOT.resolve())
        env = os.environ.copy()
        env.update(GIT_INDEX_FILE=str(temp/"index"), GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
        run(["git", "read-tree", PARENT], env=env)
        run(["git", "add", "--", *files], env=env)
        for path, contents in overrides.items():
            blob = run(["git", "hash-object", "-w", "--stdin"], stdin=contents.encode()).strip().decode()
            run(["git", "update-index", "--cacheinfo", "100644,"+blob+","+path], env=env)
        changed = run(["git", "diff", "--cached", "--name-only", PARENT], env=env).decode().splitlines()
        assert set(changed) == set(files), changed
        run(["git", "diff", "--cached", "--check", PARENT], env=env)
        tree = run(["git", "write-tree"], env=env).strip().decode()
        message = "Add "+experiment+" "+("phase-conditioned gait tracking" if route == "114" else "phase-free real-transition contrast")+" on the stable exp1.13t policy\n"
        commit = run(["git", "commit-tree", tree, "-p", PARENT], stdin=message.encode()).strip().decode()
        archive = io.BytesIO(run(["git", "archive", "--format=zip", commit, "humanoid", "tests", "resources", "scripts"]))
        checkout = temp/"checkout"
        checkout.mkdir()
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                assert (checkout/name).resolve().is_relative_to(checkout.resolve())
            source.extractall(checkout)
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=checkout, capture_output=True)
        if tests.returncode:
            raise RuntimeError(tests.stderr.decode("utf-8", "replace")[-6000:])
        preflight_file = temp/"cpu_preflight.json"
        run([sys.executable, "scripts/amp_tools/preflight_reference_transfer.py", "--route", route,
             "--source", str(ROOT/"czy/data/exp1.13t/model_6000.pt"), "--output", str(preflight_file)], cwd=checkout)
        preflight = json.loads(preflight_file.read_text())
        preflight["tested_commit"] = commit
        (directory/"cpu_preflight.json").write_text(json.dumps(preflight, indent=2)+"\n", encoding="utf-8")
        result = {"experiment": experiment, "branch": branch, "commit": commit, "parent": PARENT, "changed_files": changed,
                  "exact_isolated_tree_cpu_tests": tests.stderr.decode("utf-8", "replace").splitlines()[-4:],
                  "production_checkpoint_preflight": preflight["verified"], "published": False,
                  "original_head": original_head, "original_branch": original_branch,
                  "original_index_sha256": original_index, "created_at": datetime.now(timezone.utc).isoformat()}
        if args.push:
            run(["git", "update-ref", "refs/heads/"+branch, commit, "0"*40])
            git = ["git", "-c", "http.proxy="+PROXY, "-c", "http.sslBackend=openssl"]
            run(git+["push", "origin", "refs/heads/"+branch+":refs/heads/"+branch], env=env)
            remote = run(git+["ls-remote", "--heads", "origin", "refs/heads/"+branch], env=env).decode().split()[0]
            assert remote == commit
            result.update(published=True, remote_commit=remote, proxy=PROXY)
        assert hashlib.sha256((ROOT/".git/index").read_bytes()).hexdigest() == original_index
        assert run(["git", "rev-parse", "HEAD"]).strip().decode() == original_head
        assert run(["git", "symbolic-ref", "--short", "HEAD"]).strip().decode() == original_branch
        result["original_index_and_head_preserved"] = True
        output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
