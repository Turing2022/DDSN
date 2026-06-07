import os
from PIL import Image

def convert_gif_to_png(folder_path):
    """
    将一个文件夹中的所有 .gif 图片转换为 .png 格式并保存

    参数:
        folder_path (str): 包含 .gif 图片的文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是 .gif 格式
        if filename.lower().endswith('.gif'):
            # 获取文件的完整路径
            gif_path = os.path.join(folder_path, filename)
            # 打开 .gif 图片
            img = Image.open(gif_path)
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(filename)[0]
            # 构造新的 .png 文件名
            png_path = os.path.join(folder_path, f"{base_name}.png")
            # 保存为 .png 格式
            img.save(png_path, 'PNG')
            print(f"Converted '{filename}' to '{base_name}.png'")

# 替换为你的文件夹路径
folder_path = '/media/clc/ESD-USB/Drive/2'
convert_gif_to_png(folder_path)