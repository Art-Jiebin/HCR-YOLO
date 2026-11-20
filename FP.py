import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

# ==================== 配置区 ====================
SRC_DIR = "/root/autodl-tmp/YOLO_Dataset_Images"
OUT_DIR = "/root/autodl-tmp/YOLO_Dataset"
CLASSES = ["hair", "floc", "particle"]
RATIOS = (0.7, 0.2, 0.1)
SEED = 42
# =================================================

random.seed(SEED)
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def labelme_rect_to_yolo(json_path, img_w, img_h):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASSES:
            print(f"  [警告] 未知类别 '{label}' in {json_path}")
            continue
        cls_id = CLASSES.index(label)
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
        y_min, y_max = max(0, min(ys)), min(img_h, max(ys))
        w, h = x_max - x_min, y_max - y_min
        if w <= 1 or h <= 1:
            continue
        x_c = (x_min + x_max) / 2 / img_w
        y_c = (y_min + y_max) / 2 / img_h
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w/img_w:.6f} {h/img_h:.6f}")
    return lines


def main():
    src = Path(SRC_DIR)
    out = Path(OUT_DIR)
    if not src.exists():
        print(f"[错误] 源目录不存在: {src}")
        return

    img_files = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS)]
    print(f"共找到 {len(img_files)} 张图片")

    has_json = sum(1 for f in img_files
                   if (src / f"{Path(f).stem}.json").exists())
    print(f"有json: {has_json}   无json(负样本): {len(img_files) - has_json}")

    # 纯随机划分
    random.shuffle(img_files)
    n = len(img_files)
    n_train = int(n * RATIOS[0])
    n_val = int(n * RATIOS[1])
    splits = {
        "train": img_files[:n_train],
        "val":   img_files[n_train:n_train + n_val],
        "test":  img_files[n_train + n_val:]
    }
    print(f"\n划分: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")

    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"img": 0, "pos": 0, "neg": 0, "box": 0,
                 "cls": [0]*len(CLASSES)} for s in splits}

    for split, files in splits.items():
        print(f"\n处理 {split} ...")
        for fname in files:
            stem = Path(fname).stem
            src_img = src / fname
            src_json = src / f"{stem}.json"
            dst_img = out / "images" / split / fname
            dst_txt = out / "labels" / split / f"{stem}.txt"

            shutil.copy2(src_img, dst_img)

            lines = []
            if src_json.exists():
                try:
                    with Image.open(src_img) as im:
                        img_w, img_h = im.size
                    lines = labelme_rect_to_yolo(src_json, img_w, img_h)
                except Exception as e:
                    print(f"  [错误] {src_json}: {e}")

            if lines:
                dst_txt.write_text("\n".join(lines) + "\n", encoding='utf-8')
                stats[split]["pos"] += 1
                stats[split]["box"] += len(lines)
                for line in lines:
                    stats[split]["cls"][int(line.split()[0])] += 1
            else:
                dst_txt.touch()
                stats[split]["neg"] += 1
            stats[split]["img"] += 1

    yaml_content = f"""path: {out.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (out / "data.yaml").write_text(yaml_content, encoding='utf-8')

    print("\n" + "=" * 60)
    print("转换完成！统计：")
    for s, v in stats.items():
        print(f"\n[{s}] 图片{v['img']}  有目标{v['pos']}  负样本{v['neg']}  "
              f"总框数{v['box']}")
        for i, c in enumerate(CLASSES):
            print(f"    {c:10s} (id={i}): {v['cls'][i]} 个框")
    print(f"\ndata.yaml: {out / 'data.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()