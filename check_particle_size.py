import numpy as np
from pathlib import Path

LABEL_DIR = Path("/root/autodl-tmp/YOLO_Dataset/labels/test")
IMG_SIZE = 640

sizes_all = []
sizes_hair = []
sizes_floc = []
sizes_particle = []

CLASS_NAMES = {0: "hair", 1: "floc", 2: "particle"}

for txt in LABEL_DIR.glob("*.txt"):
    content = txt.read_text().strip()
    if not content:
        continue
    for line in content.split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        w = float(parts[3]) * IMG_SIZE
        h = float(parts[4]) * IMG_SIZE
        sizes_all.append((w, h))
        if cid == 0:
            sizes_hair.append((w, h))
        elif cid == 1:
            sizes_floc.append((w, h))
        elif cid == 2:
            sizes_particle.append((w, h))


def report(name, sizes):
    if not sizes:
        print(f"{name}: 无框")
        return
    arr = np.array(sizes)
    w, h = arr[:, 0], arr[:, 1]
    area = w * h
    print(f"\n===== {name} =====")
    print(f"框数: {len(sizes)}")
    print(f"宽  (px):  均值 {w.mean():6.1f}  中位数 {np.median(w):6.1f}  最小 {w.min():5.1f}  最大 {w.max():6.1f}")
    print(f"高  (px):  均值 {h.mean():6.1f}  中位数 {np.median(h):6.1f}  最小 {h.min():5.1f}  最大 {h.max():6.1f}")
    print(f"面积(px²): 均值 {area.mean():8.1f}  中位数 {np.median(area):8.1f}")
    # 尺寸分档统计（按最大边）
    max_side = np.maximum(w, h)
    for thresh in [8, 16, 32, 64, 128]:
        ratio = (max_side <= thresh).mean() * 100
        print(f"  最大边 <= {thresh:3d}px 占比: {ratio:5.1f}%")


print(f"标签目录: {LABEL_DIR}")
print(f"图片尺寸基准: {IMG_SIZE}x{IMG_SIZE}")

report("全部类别", sizes_all)
report("hair", sizes_hair)
report("floc", sizes_floc)
report("particle", sizes_particle)