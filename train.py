import os, time, sys
os.environ["YOLO_CONFIG_DIR"] = "/root/.config/Ultralytics"

# 注册自定义模块到 ultralytics
sys.path.insert(0, "/root/autodl-tmp")

import ultralytics.nn.modules as nn_modules
from grb_module import GRB

# 把 GRB 注册到 ultralytics 的模块表
nn_modules.GRB = GRB
# 注册到 yaml 解析器
import ultralytics.nn.tasks as tasks
tasks.GRB = GRB

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/root/autodl-tmp/yolo11n-grb.yaml").load("/root/autodl-tmp/yolo11n.pt")

    model.train(
        data="/root/autodl-tmp/YOLO_Dataset/data.yaml",
        project="/root/autodl-tmp/runs/wine_experiments",
        name=f"exp_grb_{time.strftime('%m%d_%H%M')}",
        exist_ok=True,

        epochs=150,          # ← 你要求 150
        imgsz=640,
        batch=32,            # GRB 参数量小，可以回 32
        device=0,
        workers=8,

        cls=0.8,
        copy_paste=0.3,
        mosaic=1.0,
        close_mosaic=10,
        cache=True,
        patience=0,

        plots=True,
        save=True,
        pretrained=True,
        verbose=True,
        val=True,
    )