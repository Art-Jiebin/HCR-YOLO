import os
import glob
from PIL import Image

# ================= 配置区域 =================
# 根据你的服务器目录结构，已配置好绝对路径[cite: 12]
SOURCE_DIR = r"/root/autodl-tmp/白酒杂质数据集"
OUTPUT_DIR = r"/root/autodl-tmp/YOLO_Dataset_Images"
MAX_SIZE = 1280


# ============================================

def merge_and_compress():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"正在扫描 {SOURCE_DIR} 及其子文件夹下的所有图片...")

    # 递归查找所有格式的图片
    image_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        # 排除输出目录，防止无限循环嵌套
        if os.path.abspath(root) == os.path.abspath(OUTPUT_DIR):
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("未找到任何图片，请检查 SOURCE_DIR 路径是否正确！")
        return

    print(f"共找到 {len(image_files)} 张大图，开始合并、重命名并压缩...")

    saved_size = 0
    counter = 1

    for img_path in image_files:
        # 统一重命名格式，例如 dataset_00001.jpg
        out_filename = f"dataset_{counter:05d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        orig_size = os.path.getsize(img_path)

        try:
            with Image.open(img_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # 等比例缩放计算
                width, height = img.size
                if max(width, height) > MAX_SIZE:
                    if width > height:
                        new_w = MAX_SIZE
                        new_h = int(height * (MAX_SIZE / width))
                    else:
                        new_h = MAX_SIZE
                        new_w = int(width * (MAX_SIZE / height))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # 保存并压缩
                img.save(out_path, format='JPEG', quality=85)

            new_size = os.path.getsize(out_path)
            saved_size += (orig_size - new_size)
            counter += 1

        except Exception as e:
            print(f"处理文件 {img_path} 时出错: {e}")

    print("================================")
    print("全部合并与压缩完成！")
    print(f"共处理 {counter - 1} 张图片。")
    print(f"图片已全部汇总至: {OUTPUT_DIR}")
    print(f"为你节省了大约 {saved_size / (1024 * 1024):.2f} MB 的硬盘空间！")
    print("现在你可以将新文件夹下载回本地，用 Labelme 直接从头标到尾了！")


if __name__ == '__main__':
    merge_and_compress()