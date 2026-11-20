import os
import glob
import shutil
from PIL import Image, ImageEnhance

# ================= 配置区域 =================
# 原图片所在的文件夹（你刚生成排好序的文件夹）
SOURCE_DIR = r"/root/autodl-tmp/YOLO_Dataset_Images"
# 增强后存放的新文件夹
OUTPUT_DIR = r"/root/autodl-tmp/YOLO_Dataset_Enhanced"

# --- 核心调参区 ---
# 已经根据你在 Labelme 中的视觉习惯锁定参数
BRIGHTNESS_FACTOR = 1.32
CONTRAST_FACTOR = 1.10


# ============================================

def enhance_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取所有图片
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(SOURCE_DIR, ext.upper())))

    if not image_files:
        print("未找到图片，请检查 SOURCE_DIR 路径是否正确！")
        return

    print(f"找到 {len(image_files)} 张图片，开始应用 1.32 亮度与 1.10 对比度增强...")

    json_count = 0
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        out_img_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # 1. 增强并保存图片
            with Image.open(img_path) as img:
                # 调亮度
                enhancer_brightness = ImageEnhance.Brightness(img)
                img_bright = enhancer_brightness.enhance(BRIGHTNESS_FACTOR)

                # 调对比度
                enhancer_contrast = ImageEnhance.Contrast(img_bright)
                img_final = enhancer_contrast.enhance(CONTRAST_FACTOR)

                img_final.save(out_img_path, quality=90)

            # 2. 自动迁移可能存在的 json 标签文件
            json_path = os.path.join(SOURCE_DIR, base_name + '.json')
            if os.path.exists(json_path):
                out_json_path = os.path.join(OUTPUT_DIR, base_name + '.json')
                shutil.copy(json_path, out_json_path)
                json_count += 1

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("================================")
    print("图像批量增强全部完成！")
    print(f"已迁移 {json_count} 个 JSON 标签文件。")
    print(f"你的高画质数据集已生成在: {OUTPUT_DIR}")
    print("现在可以将该文件夹打包下载回本地，开始最高效的画框标注了！")


if __name__ == '__main__':
    enhance_dataset()