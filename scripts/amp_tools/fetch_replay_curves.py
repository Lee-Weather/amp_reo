import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import concurrent.futures
import json
import flux_ops
from flux_ops import call, sanitize

def main():
    parser = argparse.ArgumentParser(description="Fetch AMP curves into an experiment archive")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account")
    parser.add_argument("--end-time", default=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"))
    args = parser.parse_args()
    if args.account:
        flux_ops.ACCOUNT = args.account
    keys = ["task_reward", "style_reward", "episode_length", "done_fraction", "expert_score", "policy_score"]
    def fetch(key):
        result = call("task", "data", "get", "--task-id", args.task_id, "--data-key", "AMP/" + key,
                      "--sampling-mode", "precise", "--end-time", args.end_time)
        return key, sanitize(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(fetch, keys))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key, result in results.items():
        points = result["data"].get("AMP/" + key, {})
        print(key, "points=", len(points.get("steps", [])), "final=", points.get("values", [])[-1:])


if __name__ == "__main__":
    main()
