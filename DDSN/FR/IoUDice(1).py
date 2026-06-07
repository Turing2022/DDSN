import os
import cv2
import numpy as np
import torch
from torchmetrics import JaccardIndex, Dice


def load_image(image_path, target_size=None):
    """
    加载图像并进行预处理
    :param image_path: 图像文件路径
    :param target_size: 目标大小 (height, width)
    :return: 预处理后的图像 (numpy array)
    """
    # 加载图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 如果需要，调整图像大小
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    # 将图像转换为二值图像（假设阈值为 127）
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 将图像值归一化到 0 和 1
    image = image.astype(np.float32) / 255.0

    return image


def calculate_metrics(pred_folder, target_folder, target_size=None):
    """
    计算 mIoU 和 mDice
    :param pred_folder: 预测图像文件夹路径
    :param target_folder: 真实图像文件夹路径
    :param target_size: 目标大小 (height, width)，如果需要调整图像大小
    :return: mIoU 和 mDice 的平均值
    """
    # 获取预测文件夹中的所有文件名
    pred_files = sorted(os.listdir(pred_folder))
    target_files = sorted(os.listdir(target_folder))

    # 初始化指标计算工具
    jaccard = JaccardIndex(task="binary")
    dice = Dice(average='micro')

    # 初始化列表来存储每个图像的指标
    miou_list = []
    mdsc_list = []

    # 遍历每个图像对
    for pred_file, target_file in zip(pred_files, target_files):
        # 构建完整的文件路径
        pred_path = os.path.join(pred_folder, pred_file)
        target_path = os.path.join(target_folder, target_file)

        # 加载图像
        pred = load_image(pred_path, target_size)
        target = load_image(target_path, target_size)

        # 确保预测和目标图像大小相同
        if pred.shape != target.shape:
            print(f"Warning: Image size mismatch for {pred_file} and {target_file}. Skipping...")
            continue

        # 将 numpy 数组转换为 PyTorch 张量
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

        # 计算 mIoU 和 mDice
        miou = jaccard(pred_tensor, target_tensor)
        mdsc = dice(pred_tensor, target_tensor)

        # 存储结果
        miou_list.append(miou.item())
        mdsc_list.append(mdsc.item())

    # 计算平均值
    avg_miou = np.mean(miou_list) if miou_list else 0.0
    avg_mdsc = np.mean(mdsc_list) if mdsc_list else 0.0

    return avg_miou, avg_mdsc


# 示例用法
if __name__ == "__main__":
    # 设置预测和真实图像文件夹路径

    pred_folder = "/media/clc/ESD-USB/Drive/degrade"  # 预测结果文件夹
    target_folder = "/media/clc/ESD-USB/Drive/2"  # 真实标签文件夹
    # 可选：设置目标图像大小
    target_size = (512, 512)  # 如果需要调整图像大小，否则设为 None

    # 计算 mIoU 和 mDice
    avg_miou, avg_mdsc = calculate_metrics(pred_folder, target_folder, target_size)

    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average mDice: {avg_mdsc:.4f}")