import os

def rename_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是图片（可以根据需要扩展图片格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 获取文件的完整路径
            old_file = os.path.join(folder_path, filename)
            # 如果文件名长度大于10，删除前10个字符
            if len(filename) > 10:
                new_filename = filename[10:]
                new_file = os.path.join(folder_path, new_filename)
                # 重命名文件
                os.rename(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'")

# 替换为你的文件夹路径
folder_path = '/media/clc/ESD-USB/Drive/degrade'
rename_images(folder_path)