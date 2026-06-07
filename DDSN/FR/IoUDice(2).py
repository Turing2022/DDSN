import os
import cv2
import numpy as np
import torch


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


def calculate_miou(pred, target):
    """
    计算 mIoU
    :param pred: 预测图像 (numpy array)，形状为 [H, W]
    :param target: 真实图像 (numpy array)，形状为 [H, W]
    :return: mIoU 值
    """
    # 将 numpy 数组转换为 PyTorch 张量
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    # 计算交集和并集
    intersection = (pred * target).sum().float()
    union = (pred + target).sum().float() - intersection

    # 计算 IoU
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()


def calculate_mdice(pred, target):
    """
    计算 mDice
    :param pred: 预测图像 (numpy array)，形状为 [H, W]
    :param target: 真实图像 (numpy array)，形状为 [H, W]
    :return: mDice 值
    """
    # 将 numpy 数组转换为 PyTorch 张量
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    # 计算交集和并集
    intersection = (pred * target).sum().float()
    union = pred.sum().float() + target.sum().float()

    # 计算 Dice 系数
    dice = (2. * intersection + 1e-6) / (union + 1e-6)

    return dice.item()


def calculate_metrics(pred_folder, target_folder, target_size=None):
    """
    计算 mIoU 和 mDice
    :param pred_folder: 预测图像文件夹路径
    :param target_folder: 真实图像文件夹路径
    :param target_size: 目标大小 (height, width)，如果需要调整图像大小
    :return: 平均 mIoU 和 mDice 值
    """
    # 获取预测文件夹中的所有文件名
    pred_files = sorted(os.listdir(pred_folder))
    target_files = sorted(os.listdir(target_folder))

    # 初始化列表来存储每个图像的指标
    miou_list = []
    mdice_list = []

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

        # 计算 mIoU 和 mDice
        miou = calculate_miou(pred, target)
        mdice = calculate_mdice(pred, target)

        # 存储结果
        miou_list.append(miou)
        mdice_list.append(mdice)

    # 计算平均值
    avg_miou = np.mean(miou_list) if miou_list else 0.0
    avg_mdice = np.mean(mdice_list) if mdice_list else 0.0

    return avg_miou, avg_mdice


if __name__ == "__main__":
    # 设置预测和真实图像文件夹路径
    pred_folder = "/home/clc/公共的/wyf/FR-UNet-master/ours"  # 替换为你的预测图像文件夹路径
    target_folder = "/media/clc/ESD-USB/Drive/2"  # 替换为你的真实图像文件夹路径

    # 可选：设置目标图像大小
    target_size = (512, 512)  # 如果需要调整图像大小，否则设为 None

    # 计算 mIoU 和 mDice
    avg_miou, avg_mdice = calculate_metrics(pred_folder, target_folder, target_size)

    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average mDice: {avg_mdice:.4f}")