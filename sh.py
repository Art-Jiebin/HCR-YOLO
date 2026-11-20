import os, json
from pathlib import Path

SRC = "/root/autodl-tmp/YOLO_Dataset_Images"   # ← 改成你的路径
labels = {}
for f in os.listdir(SRC):
    if f.endswith(".json"):
        try:
            with open(Path(SRC)/f, encoding='utf-8') as fp:
                d = json.load(fp)
            for s in d.get("shapes", []):
                labels[s["label"]] = labels.get(s["label"], 0) + 1
        except Exception as e:
            print(f"跳过 {f}: {e}")

print("所有出现过的标签：")
for k, v in sorted(labels.items(), key=lambda x: -x[1]):
    print(f"  {k!r}: {v} 次")