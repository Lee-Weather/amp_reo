import json, base64

# Fresh download URL
url = "https://limx-gmotion.oss-cn-beijing.aliyuncs.com/placeholder/EXPIRED_URL"

# Get fresh from API
import subprocess
result = subprocess.run(
    ["flux", "task", "model", "list", "--task-id", "TASK_20261111_106", "--checkpoint", "11950", "--limit", "1"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
url = data["data"]["rows"][0]["policUrlDown"]
print(f"URL: {url[:80]}...")

# We can't use wget/curl because the URL has & characters
# Use base64 approach - embed the download in Python
b64 = base64.urlsafe_b64encode(url.encode()).decode()

payload = {
    "taskBaseInfo": {
        "projectId": "PRO_20261111_001",
        "taskType": "1",
        "trainType": "1",
        "taskName": "exp3.3-pb6",
        "taskDescription": "exp3.3: playback model_11950 via python download",
        "goodsId": "ESKU000001",
        "imageId": "BJX00000001",
        "imageVersion": "V000412"
    },
    "taskCodeInfo": {
        "codeUrl": '[{"codeUrl":"https://github.com/Lee-Weather/amp_reo.git","versionType":"3","versionName":"main"}]',
        "codeType": "2",
        "startScript": f"cd amp_reo && python -c \"import urllib.request;urllib.request.urlretrieve('https://bit.ly/3Dummy','m.pt')\" && python humanoid/scripts/play_amp.py --task=x1_amp_exp33 --headless --checkpoint_path=m.pt",
        "isOpen": "1",
        "mainCodeUri": "",
        "hparamsPath": ""
    }
}

with open("cr.json", "w") as f:
    json.dump(payload, f, indent=2)

print("Written cr.json")
print(json.dumps(payload, indent=2))
