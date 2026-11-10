import json
b64 = "aHR0cHM6Ly9saW14LWdyYWRtb3Rpb24ub3NzLWNuLWJlaWppbmcuYWxpeXVuY3MuY29tL3VwbG9hZCUyRjIwMjYlMkYxMSUyRjklMkZtb2RlbF8xMTc1MF8yMDI2MTEwOTExMzAxNUE1MzQucHQ/T1NTQWNjZXNzS2V5SWQ9TFRBSTV0OTFxV3VHUXEyU2hjSGNGcVd2JkV4cGlyZXM9MTc5NDg5NDExNSZTaWduYXR1cmU9TWZpREZDTmh1V3ppcENWVmxCT1ZQeXRXQ2QwJTNE"
task = {
  "taskBaseInfo": {
    "projectId": "PRO_20261109_001",
    "taskType": "1",
    "trainType": "1",
    "taskName": "exp3.3-standard-upright-v10",
    "goodsId": "ESKU000001",
    "imageId": "BJX00000001",
    "imageVersion": "V000412"
  },
  "taskCodeInfo": {
    "codeType": "2",
    "codeUrl": '[{"codeUrl":"https://github.com/Lee-Weather/amp_reo.git","versionType":"3","versionName":"9f7f0d3a3ee7639cff25cc7028b65fe7ea503df4"}]',
    "isOpen": "1",
    "startScript": f"python /workspace/amp_reo/humanoid/scripts/train.py --task=x1_amp_exp33 --headless --num_envs=4096 --seed=5 --max_iterations=200 --run_name=exp3.3_main --warm_start=/workspace/amp_reo/logs/x1_amp_exp33/exported_data/input_model_11750.pt --warm_start_sha256=391f83466f26c76c0949f68b1771e01b3b4c5a49642527f1f3aea1413bafc --warm_start_url_b64={b64}",
    "hparamsPath": ""
  },
  "runtimeReminderConfig": {
    "enableRuntimeReminder": True,
    "runtimeReminderMinute": 60,
    "reminderDurations": [60, 120, 180]
  }
}
with open("e:/X1/amp_re0/clean_loop/task_v10.json", "w") as f:
    json.dump(task, f, indent=2)
print("Written OK")
