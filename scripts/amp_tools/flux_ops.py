"""Local Flux helper. Never persist or print account credentials."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "limxmts0chns5mnse9@uberip.com"
CLI = Path(os.environ["APPDATA"])/"npm/node_modules/@limxdynamics/flux-cli/vendor/flux.exe"


def credentials():
    with (ROOT/"czy/skills/flux-cli/api_key.json").open(encoding="utf-8-sig") as stream:
        return next(a for a in json.load(stream)["accounts"] if a["email"] == ACCOUNT)


def sanitize(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if re.search(r"(?i)token|secret|password|api.?key|startscript|urlDown|logoUrl", k)
                    else sanitize(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9_-]+)", "[REDACTED]", value)
        value = re.sub(r"https?://[^\s\"<>]+\?[^\s\"<>]+", "[SIGNED_URL_REDACTED]", value)
    return value


def call(*args):
    env = os.environ.copy()
    env.update(FLUX_API_KEY=credentials()["api_key"], FLUX_BASE_URL="https://internal.limxdynamics.com/dev-api")
    proc = subprocess.run([str(CLI), "--yes", "--timeout", "30s", "--retry", "1", *args],
                          env=env, capture_output=True, encoding="utf-8", errors="replace")
    try:
        data = json.loads(proc.stdout if proc.returncode in (0, 10) else proc.stderr)
    except ValueError:
        raise RuntimeError("Flux returned non-JSON, exit="+str(proc.returncode)) from None
    if proc.returncode not in (0, 10):
        raise RuntimeError(json.dumps(sanitize(data), ensure_ascii=False))
    return data


def github_get(path):
    token = credentials()["github_token"]
    config = 'header = "Authorization: Bearer '+token+'"\nheader = "Accept: application/vnd.github+json"\n'
    p = subprocess.run(["C:/Windows/System32/curl.exe", "-sS", "--ssl-no-revoke", "--fail-with-body", "--connect-timeout", "15", "--max-time", "60",
                        "--proxy", "socks5h://10.12.201.122:39009", "--config", "-",
                        "https://api.github.com/"+path], input=config, capture_output=True, encoding="utf-8", errors="replace")
    if not p.stdout:
        raise RuntimeError("GitHub connection failed, exit="+str(p.returncode))
    d = json.loads(p.stdout)
    if p.returncode:
        raise RuntimeError("GitHub request failed: "+str(d.get("message")))
    return d


if __name__ == "__main__":
    if sys.argv[1:] in (["--help"], ["-h"]):
        print("Usage: python scripts/amp_tools/flux_ops.py <Flux CLI arguments>\n"
              "Credentials are read from the existing ignored api_key.json; output is sanitized.")
    elif sys.argv[1:] == ["inventory"]:
        me = call("auth", "whoami")["data"]
        print(json.dumps({"identity": {"userId": me["userId"], "email": me["email"],
                                     "git_configured": bool(me.get("githubToken"))}}, ensure_ascii=False))
        for args in [("task", "image", "versions", "--image-id", "BJX00000001"),
                     ("task", "info", "--task-id", "TASK_20260908_181")]:
            d = call(*args)
            if args[1] == "info":
                body = d["data"]
                print("TASK_INFO_ROOT_FIELDS",list(body) if isinstance(body,dict) else type(body).__name__)
                print(json.dumps(sanitize(body), ensure_ascii=False))
            else:
                print(json.dumps(sanitize(d), ensure_ascii=False))
        repo = github_get("repos/Lee-Weather/amp_reo")
        print(json.dumps({"repository":repo["full_name"], "private":repo["private"],
                          "default_branch":repo["default_branch"], "push_permission":repo.get("permissions",{}).get("push")},ensure_ascii=False))
    else:
        print(json.dumps(sanitize(call(*sys.argv[1:])),ensure_ascii=False))
