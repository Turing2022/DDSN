import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import vessel_dataset
import models
from utils.helpers import get_instance
from bunch import Bunch
from ruamel.yaml import YAML


def debug_test():
    """调试版本，直接处理原始图像"""
    data_path = "/home/clc/公共的/wyf/FR-UNet-master/DRIVE"
    weight_path = "pretrained_weights/DRIVE/checkpoint-epoch40.pth"
    output_dir = "segmentation_results"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置和模型
    yaml = YAML(typ='safe', pure=True)
    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(yaml.load(file))

    checkpoint = torch.load(weight_path, weights_only=False)
    model = get_instance(models, 'model', CFG)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 直接处理测试图像文件夹中的文件
    test_images_dir = os.path.join(data_path, "test", "images")
    image_files = [f for f in os.listdir(test_images_dir)
                   if f.lower().endswith('.png')]

    print(f"开始处理 {len(image_files)} 张图像...")

    # 这里需要根据你的数据集类来创建适当的数据转换
    # 由于不清楚具体的数据预处理，这里提供一个简化版本
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 添加其他必要的预处理
    ])

    for image_file in image_files:
        try:
            # 读取图像
            image_path = os.path.join(test_images_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"无法读取图像: {image_file}")
                continue

            # 预处理图像（根据你的模型要求调整）
            image_tensor = torch.from_numpy(image).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, H, W]

            # 模型预测
            with torch.no_grad():
                pred = model(image_tensor)
                pred = pred[0, 0, ...]  # 取第一个样本的第一个通道
                predict = torch.sigmoid(pred).cpu().numpy()
                predict_binary = np.where(predict >= 0.5, 1, 0)

            # 保存分割图，使用原文件名
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, np.uint8(predict_binary * 255))
            print(f"已生成: {image_file}")

        except Exception as e:
            print(f"处理 {image_file} 时出错: {str(e)}")

    print(f"完成! 分割图保存在: {output_dir}")


if __name__ == '__main__':
    debug_test()