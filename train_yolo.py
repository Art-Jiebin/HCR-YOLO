import os
os.environ["YOLO_CONFIG_DIR"] = "/root/.config/Ultralytics"

from ultralytics import YOLO

if __name__ == "__main__":
    # ============ 实验配置区（每次只改这里） ============
    EXP_NAME = "yolov5n"   # 本次实验名，
    MODEL_PATH = "/root/autodl-tmp/yolov5n.pt"
    DATA_YAML  = "/root/autodl-tmp/YOLO_Dataset/data.yaml"
    PROJECT_DIR = "/root/autodl-tmp/runs/wine_experiments"   # 大文件夹
    # ==================================================

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=32,
        device=0,

        # ===== 目录管理核心 =====
        project=PROJECT_DIR,   # 所有实验的大文件夹
        name=EXP_NAME,         # 本次实验的子文件夹
        exist_ok=False,        # 同名文件夹存在就报错，避免误覆盖
        # ======================

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