import os
import numpy as np
import cv2


def calculate_metrics(pred_folder, label_folder, num_classes=None, ignore_index=None):
    """
    计算图片分割任务的mIoU和mDice指标

    参数:
        pred_folder (str): 预测结果图片的文件夹路径
        label_folder (str): 真实标签图片的文件夹路径
        num_classes (int, optional): 类别数量，若为None则自动检测
        ignore_index (int, optional): 需要忽略的类别索引（如背景）

    返回:
        tuple: (mIoU, mDice)
    """
    # 自动检测类别数量
    if num_classes is None:
        max_cls = 0
        for folder in [pred_folder, label_folder]:
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"无法读取图像：{img_path}")
                max_cls = max(max_cls, np.max(img))
        num_classes = max_cls + 1
        print(f"自动检测到 {num_classes} 个类别")

    # 初始化统计量
    tp = np.zeros(num_classes, dtype=np.uint64)
    fp = np.zeros(num_classes, dtype=np.uint64)
    fn = np.zeros(num_classes, dtype=np.uint64)

    # 获取并检查文件列表
    pred_files = sorted(os.listdir(pred_folder))
    label_files = sorted(os.listdir(label_folder))
    if pred_files != label_files:
        raise ValueError("文件名不匹配，请确保预测和标签文件名一致")

    # 遍历每对图像
    for pred_file, label_file in zip(pred_files, label_files):
        pred_path = os.path.join(pred_folder, pred_file)
        label_path = os.path.join(label_folder, label_file)

        # 读取图像
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or label is None:
            raise ValueError(f"图像读取失败：{pred_path} 或 {label_path}")

        # 检查尺寸一致性
        if pred.shape != label.shape:
            raise ValueError(f"图像尺寸不一致：{pred_file} vs {label_file}")

        # 更新统计量
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            label_cls = (label == cls)
            tp[cls] += np.sum(np.logical_and(pred_cls, label_cls))
            fp[cls] += np.sum(np.logical_and(pred_cls, ~label_cls))
            fn[cls] += np.sum(np.logical_and(~pred_cls, label_cls))

    # 计算各类别指标
    iou_list = []
    dice_list = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        union = tp[cls] + fp[cls] + fn[cls]
        denominator_dice = (2 * tp[cls]) + fp[cls] + fn[cls]

        iou = tp[cls] / union if union != 0 else np.nan
        dice = (2 * tp[cls]) / denominator_dice if denominator_dice != 0 else np.nan

        iou_list.append(iou)
        dice_list.append(dice)

    # 计算均值（忽略NaN值）
    miou = np.nanmean(iou_list)
    mdice = np.nanmean(dice_list)

    return miou, mdice


if __name__ == "__main__":
    # 使用示例
    pred_folder = "/media/clc/ESD-USB/Drive/degrade"  # 预测结果文件夹
    label_folder = "/media/clc/ESD-USB/Drive/2"  # 真实标签文件夹
    ignore_index = 0  # 忽略背景类别（可选）

    miou, mdice = calculate_metrics(pred_folder, label_folder, ignore_index=ignore_index)

    print(f"mIoU: {miou:.4f}")
    print(f"mDice: {mdice:.4f}")