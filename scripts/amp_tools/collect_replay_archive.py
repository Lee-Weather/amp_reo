"""Collect and verify a finished six-stage AMP replay without temporary clutter."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
HOST = "robot@10.12.201.5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    artifacts = ROOT / "czy/exp1/artifacts" / args.experiment
    state = json.loads((artifacts / "replay_job.json").read_text())
    remote = state["server_dir"]
    result = subprocess.run([SSH, "-o", "BatchMode=yes", HOST, "tail -n 1 " + remote + "/replay.log"],
                            check=True, capture_output=True, text=True)
    if "Replay files closed: 1500 frames" not in result.stdout:
        raise RuntimeError("Replay has not finished successfully; inspect the existing job, do not restart")
    destination = ROOT / "czy/data" / args.experiment
    if destination.exists():
        raise FileExistsError("Preserve existing replay: " + str(destination))
    with tempfile.TemporaryDirectory(prefix="x1_amp_collect_") as temp:
        temp = Path(temp)
        for name in ("isaac_diag.csv", "play_output.mp4", "replay_metadata.json"):
            subprocess.run([SCP, "-o", "BatchMode=yes", HOST + ":" + remote + "/logs/x1_amp/play_output/latest_play/" + name, str(temp / name)], check=True)
        subprocess.run([SCP, "-o", "BatchMode=yes", HOST + ":" + remote + "/replay.log", str(temp / "replay.log")], check=True)
        metadata = json.loads((temp / "replay_metadata.json").read_text())
        assert metadata["captured_steps"] == 3000 and metadata["checkpoint"] == 6000
        assert metadata["video_frames"] == 1500 and len(metadata["segments"]) == 6
        with (temp / "isaac_diag.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3000
        numeric = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        assert all(np.isfinite(values).all() for values in numeric.values())
        assert np.max(np.abs(numeric["command_applied"] - numeric["command_observed"])) < 1e-6
        assert set(numeric["active_stage"].astype(int)) == set(range(6))
        video = cv2.VideoCapture(str(temp / "play_output.mp4"))
        assert video.isOpened()
        assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1920, 1080)
        fps = video.get(cv2.CAP_PROP_FPS)
        assert abs(fps - 25) < .001
        frames = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            assert frame.std() > 5
            frames += 1
        video.release()
        assert frames == 1500
        model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        assert model_hash == state["checkpoint_sha256"]
        done = numeric["done"].astype(bool)
        timeout = numeric["timeout"].astype(bool)
        summary = dict(rows=len(rows), duration_s=float(numeric["time_s"][-1]), decoded_frames=frames,
                       video_fps=fps, video_resolution=[1920, 1080], resets=int(done.sum()),
                       timeouts=int((done & timeout).sum()), non_timeout_terminations=int((done & ~timeout).sum()),
                       command_observation_max_error=float(np.max(np.abs(numeric["command_applied"]-numeric["command_observed"]))),
                       requested_applied_different_steps=int(np.count_nonzero(np.abs(numeric["command_requested"]-numeric["command_applied"]) > 1e-6)),
                       mean_task_reward=float(numeric["task_reward"].mean()),
                       mean_style_reward=float(numeric["style_reward"].mean()),
                       torque_clip_count=int(numeric["clip_count"][-1]), checkpoint_sha256=model_hash,
                       segments=metadata["segments"])
        destination.mkdir(parents=True)
        shutil.copy2(args.model, destination / "model_6000.pt")
        for name in ("isaac_diag.csv", "play_output.mp4"):
            shutil.copy2(temp / name, destination / name)
        for name in ("replay_metadata.json", "replay.log"):
            shutil.copy2(temp / name, artifacts / name)
        summary["file_bytes"] = {path.name: path.stat().st_size for path in destination.iterdir()}
        (artifacts / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert sorted(p.name for p in destination.iterdir()) == ["isaac_diag.csv", "model_6000.pt", "play_output.mp4"]
    note = ROOT / "czy/exp1/exp1.md"
    text = note.read_text(encoding="utf-8")
    text = text.replace("完整回放结果待生成后补充。", "完整回放已完成，结果如下。")
    text += "\n| 段 | 请求vx | 实际施加vx均值 | 实际vx均值 | 跟踪RMSE | 重置（超时） |\n| --- | --- | --- | --- | --- | --- |\n"
    for segment in summary["segments"]:
        text += "| {stage} | {requested:.2f} | {average_command:.3f} | {average_vx:.3f} | {rmse_vx:.3f} | {resets}（{timeouts}） |\n".format(**segment)
    text += "\n验证：CSV共3000行、全部数值有限，观测与施加指令最大误差{:.2g}；MP4为1920×1080、25fps、1500帧，全部逐帧解码通过。共{}次重置，其中{}次超时、{}次非超时终止。平均任务奖励{:.6f}、平均AMP风格奖励{:.6f}（均为混合前）。\n".format(summary["command_observation_max_error"], summary["resets"], summary["timeouts"], summary["non_timeout_terminations"], summary["mean_task_reward"], summary["mean_style_reward"])
    text += "\n最新三件套：`czy/data/{}/`，仅model_6000.pt、play_output.mp4、isaac_diag.csv。回放元数据、日志、摘要及部署信息放在 `czy/exp1/artifacts/{}/`。模型与旧版相同，SHA256为`{}`。服务器目录：`{}`。\n".format(args.experiment, args.experiment, model_hash, remote)
    text += "\n**本次回放结论：⚠️部分达标。** 三件套交付与完整流程验证完成。步态质量应按上表分段比较指令与实际速度；AMP风格奖励仍偏低，不能视为参考步态已收敛。未新增训练，旧版固定速度回放结果保留，和最新分段结果不直接混算。\n"
    note.write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
