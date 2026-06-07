import os

def rename_files_in_folder(folder_path):
    """
    将一个文件夹中所有文件的名称前十位字符删除

    参数:
        folder_path (str): 文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名长度是否大于10
        if len(filename) > 10:
            # 获取文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            # 删除前10个字符
            new_filename = filename[10:]
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

# 替换为你的文件夹路径
folder_path = '/home/clc/公共的/wyf/FR-UNet-master/ours'
rename_files_in_folder(folder_path)