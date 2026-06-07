import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torch.backends.cudnn as cudnn
import time
from tqdm import tqdm

# 导入你的模型
from Best_module.BEE1 import net


class SimpleFolderTester:
    def __init__(self):
        # 在这里直接设置参数
        self.weights_path = "/home/clc/公共的/wyf/DPEC-VM-main/modelsqkT=4ldr/VMLL_LOLv1_pre/best.pth"   # 修改为你的权重路径
        self.input_folder = "/home/clc/公共的/wyf/DPEC-VM-main/output/VMLL_LOLv1_pre/eyeq"  # 修改为你的输入文件夹
        self.output_folder = "/home/clc/公共的/wyf/DPEC-VM-main/output/VMLL_LOLv1_pre/snneyeq"  # 修改为你的输出文件夹
        self.device_id = "0"  # GPU ID

        # 自动设置设备
        self.device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = self.load_model()
        self.transform = ToTensor()

        if torch.cuda.is_available():
            cudnn.benchmark = True

    def load_model(self):
        """加载模型和权重"""
        print(f"正在加载模型: {self.weights_path}")

        model = net()

        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location='cpu')

            # 处理不同的 checkpoint 格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 修复 key 不匹配问题
            model_state_dict = model.state_dict()
            new_state_dict = {}
            for key in state_dict:
                # 去掉多余的 'body.' 前缀
                new_key = key.replace('body.', '')
                if new_key in model_state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    print(f"跳过无法匹配的 key: {key}")

            model_state_dict.update(new_state_dict)
            model.load_state_dict(model_state_dict, strict=False)

            print("模型权重加载成功!")
        else:
            raise FileNotFoundError(f"权重文件不存在: {self.weights_path}")

        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        """预处理单张图像"""
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)

            # 调整大小为8的倍数
            new_width = original_size[0] - (original_size[0] % 8)
            new_height = original_size[1] - (original_size[1] % 8)

            resized_image = image.resize((new_width, new_height), Image.BILINEAR)

            # 转换为 tensor
            tensor_image = self.transform(resized_image).unsqueeze(0).to(self.device)

            return tensor_image, original_size, resized_image.size

        except Exception as e:
            print(f"预处理失败 {os.path.basename(image_path)}: {str(e)}")
            return None, None, None

    def enhance_image(self, input_tensor):
        """增强图像"""
        with torch.no_grad():
            enhanced_tensor = self.model(input_tensor)
        return enhanced_tensor

    def postprocess_image(self, output_tensor, original_size, resized_size):
        """后处理输出图像"""
        # 转换为 numpy 数组
        output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

        # 转换回 PIL 图像
        output_image = Image.fromarray(output_np)

        # 调整回原始尺寸
        if resized_size != original_size:
            output_image = output_image.resize(original_size, Image.BILINEAR)

        return output_image

    def process_single_image(self, input_path, output_path):
        """处理单张图像"""
        try:
            # 预处理
            input_tensor, original_size, resized_size = self.preprocess_image(input_path)
            if input_tensor is None:
                return False

            # 增强
            enhanced_tensor = self.enhance_image(input_tensor)

            # 后处理
            enhanced_image = self.postprocess_image(enhanced_tensor, original_size, resized_size)

            # 保存
            enhanced_image.save(output_path)
            return True

        except Exception as e:
            print(f"处理失败 {os.path.basename(input_path)}: {str(e)}")
            return False

    def run_test(self):
        """运行测试"""
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                            '.JPG', '.JPEG', '.PNG', '.BMP'}

        # 检查输入文件夹
        if not os.path.exists(self.input_folder):
            print(f"错误: 输入文件夹不存在 - {self.input_folder}")
            return

        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)

        # 获取所有图像文件
        image_files = []
        for file in os.listdir(self.input_folder):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file)

        if not image_files:
            print(f"在文件夹中没有找到图像文件: {self.input_folder}")
            print(f"支持的格式: {image_extensions}")
            return

        print(f"找到 {len(image_files)} 张图像")
        print(f"输入文件夹: {self.input_folder}")
        print(f"输出文件夹: {self.output_folder}")
        print("开始处理...")

        # 处理图像
        success_count = 0
        start_time = time.time()

        for filename in tqdm(image_files, desc="处理进度"):
            input_path = os.path.join(self.input_folder, filename)

            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}{ext}"
            output_path = os.path.join(self.output_folder, output_filename)

            # 处理图像
            if self.process_single_image(input_path, output_path):
                success_count += 1

        # 统计信息
        total_time = time.time() - start_time
        avg_time = total_time / len(image_files) if image_files else 0

        print(f"\n✅ 处理完成!")
        print(f"成功处理: {success_count}/{len(image_files)} 张图像")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每张: {avg_time:.2f} 秒")
        print(f"结果保存在: {self.output_folder}")


# 直接运行
if __name__ == '__main__':
    # 创建测试器并运行
    tester = SimpleFolderTester()
    tester.run_test()