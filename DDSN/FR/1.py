from PIL import Image
import os

# 设置图片目录和输出目录
input_directory = "/home/clc/公共的/wyf/FR-UNet-master/DRIVE/test/1st_manual"  # 替换为你的图片目录
output_directory = "/home/clc/公共的/wyf/FR-UNet-master/DRIVE/test/1st_manual"  # 替换为你想保存修改后图片的目录

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取目录中所有的图片文件
for filename in os.listdir(input_directory):
    # 只处理图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # 打开图片
        with Image.open(image_path) as img:
            # 修改图片大小为 256x256
            img_resized = img.resize((512, 512))

            # 保存修改后的图片
            img_resized.save(output_path)
            print(f"图片 {filename} 已调整为 256x256 并保存到 {output_path}")
