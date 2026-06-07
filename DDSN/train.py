import argparse
import itertools
import time
from os import listdir
from os.path import join
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

import lib.pytorch_ssim as pytorch_ssim
# from loss_ssim import SSIMa
from lib.data import get_training_set, is_image_file
from lib.utils import TVLoss, print_network, VGGPerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from DDSN import net
import warnings
from os.path import join, isfile

# 添加分割相关的导入
from bunch import Bunch
from ruamel.yaml import YAML
import segmodel
from utils.helpers import get_instance

# from interfence_seg import seg_main,parse_args
warnings.filterwarnings('ignore')
import logging
import os
import random

# Using model_LOL by default

masktensor_dict = {}


def histogram_loss(tensor1, tensor2):
    # 将张量移回到主机内存，并且不要求梯度
    tensor1_cpu = tensor1.detach().cpu().numpy()
    tensor2_cpu = tensor2.detach().cpu().numpy()
    # 将张量展平为一维数组
    flat_tensor1 = tensor1_cpu.flatten()
    flat_tensor2 = tensor2_cpu.flatten()

    # 计算直方图
    hist_tensor1, _ = np.histogram(flat_tensor1, bins=256, range=[0, 1])
    hist_tensor2, _ = np.histogram(flat_tensor2, bins=256, range=[0, 1])

    # 归一化直方图
    hist_tensor1 = hist_tensor1 / np.sum(hist_tensor1)
    hist_tensor2 = hist_tensor2 / np.sum(hist_tensor2)

    # 计算直方图差异度
    loss = 0.5 * np.sum(np.abs(hist_tensor1 - hist_tensor2))

    return loss


# 添加分割损失计算函数
class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    def forward(self, pred, target):
        """计算IoU和Dice损失"""
        # 二值化预测
        pred_binary = torch.sigmoid(pred)

        # 计算IoU损失
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou

        # 计算Dice损失
        dice = (2. * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)
        dice_loss = 1 - dice

        return iou_loss, dice_loss


def load_segmentation_model(weight_path, config_path="config.yaml"):
    """加载预训练的分割模型"""
    yaml = YAML(typ='safe', pure=True)
    with open(config_path, encoding="utf-8") as file:
        CFG = Bunch(yaml.load(file))

    checkpoint = torch.load(weight_path, weights_only=False)
    model = get_instance(segmodel, 'model', CFG)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def generate_segmentation_mask(model, image_tensor):
    """为增强后的图像生成分割掩码"""
    with torch.no_grad():
        # 假设模型输入是单通道灰度图，将RGB转为灰度
        if image_tensor.shape[1] == 3:  # RGB图像
            # 使用标准RGB转灰度公式
            gray_tensor = 0.299 * image_tensor[:, 0:1, :, :] + \
                          0.587 * image_tensor[:, 1:2, :, :] + \
                          0.114 * image_tensor[:, 2:3, :, :]
        else:
            gray_tensor = image_tensor

        # 归一化到[0, 1]
        gray_tensor = (gray_tensor - gray_tensor.min()) / (gray_tensor.max() - gray_tensor.min() + 1e-8)

        # 模型预测
        pred = model(gray_tensor)
        pred = pred[:, 0, ...]  # 取第一个通道
        predict = torch.sigmoid(pred)

        return predict


# 自定义数据集类，支持加载分割标签
class EnhancedDataset(Dataset):
    def __init__(self, data_root, augment=True, patch_size=256):
        super(EnhancedDataset, self).__init__()
        self.augment = augment
        self.patch_size = patch_size

        # 图像路径
        self.low_light_dir = os.path.join(data_root, "low")
        self.normal_light_dir = os.path.join(data_root, "high")
        self.seg_mask_dir = os.path.join(data_root, "seg")  # 分割标签目录

        # 获取图像文件名
        self.low_light_images = [f for f in sorted(os.listdir(self.low_light_dir))
                                 if is_image_file(f)]
        self.normal_light_images = [f for f in sorted(os.listdir(self.normal_light_dir))
                                    if is_image_file(f)]
        self.seg_masks = [f for f in sorted(os.listdir(self.seg_mask_dir))
                          if is_image_file(f)]
        print(len(self.low_light_images))
        print(len(self.normal_light_images))
        print(len(self.seg_masks))
        # 确保文件数量一致
        assert len(self.low_light_images) == len(self.normal_light_images) == len(self.seg_masks), \
            "Number of low-light, normal-light and segmentation mask images should be the same"

        self.transform = ToTensor()

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        # 读取低光图像
        low_light_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        low_light_img = Image.open(low_light_path).convert('RGB')

        # 读取正常光图像
        normal_light_path = os.path.join(self.normal_light_dir, self.normal_light_images[idx])
        normal_light_img = Image.open(normal_light_path).convert('RGB')

        # 读取分割标签
        seg_mask_path = os.path.join(self.seg_mask_dir, self.seg_masks[idx])
        seg_mask = Image.open(seg_mask_path).convert('L')  # 灰度模式读取

        # 数据增强
        if self.augment:
            # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(
                low_light_img, output_size=(self.patch_size, self.patch_size))

            low_light_img = transforms.functional.crop(low_light_img, i, j, h, w)
            normal_light_img = transforms.functional.crop(normal_light_img, i, j, h, w)
            seg_mask = transforms.functional.crop(seg_mask, i, j, h, w)

            # 随机水平翻转
            if random.random() > 0.5:
                low_light_img = transforms.functional.hflip(low_light_img)
                normal_light_img = transforms.functional.hflip(normal_light_img)
                seg_mask = transforms.functional.hflip(seg_mask)

            # 随机垂直翻转
            if random.random() > 0.5:
                low_light_img = transforms.functional.vflip(low_light_img)
                normal_light_img = transforms.functional.vflip(normal_light_img)
                seg_mask = transforms.functional.vflip(seg_mask)
        else:
            # 中心裁剪
            low_light_img = transforms.CenterCrop(self.patch_size)(low_light_img)
            normal_light_img = transforms.CenterCrop(self.patch_size)(normal_light_img)
            seg_mask = transforms.CenterCrop(self.patch_size)(seg_mask)

        # 转换为Tensor
        low_light_tensor = self.transform(low_light_img)
        normal_light_tensor = self.transform(normal_light_img)
        seg_mask_tensor = torch.from_numpy(np.array(seg_mask)).float() / 255.0
        seg_mask_tensor = seg_mask_tensor.unsqueeze(0)  # 添加通道维度

        return low_light_tensor, normal_light_tensor, seg_mask_tensor


def cfg():
    parser = argparse.ArgumentParser(description='low-light image enhancement by SMNet')
    parser.add_argument('--trainset', type=str, default='/home/clc/公共的/wyf/DPEC-VM-main/datasets/train',
                        help='location of trainset')
    parser.add_argument('--testset', type=str, default='/home/clc/公共的/wyf/DPEC-VM-main/datasets/test',
                        help='location of testset')
    parser.add_argument('--output', default='output', help='location to save output images')
    parser.add_argument('--modelname', default='VMLL_LOLv1_pre_APechange', help='define model name')
    parser.add_argument('--deviceid', default="0", help='selecte which gpu device')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument('--lr_min', type=float, default=5e-5, help='Minimum Learning Rate')
    parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=24, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=0, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped LR image')

    # 添加分割相关的参数
    parser.add_argument('--seg_weight_path', type=str, default="/home/clc/公共的/wyf/DPEC-VM-main/FR/pretrained_weights/DRIVE/checkpoint-epoch40.pth",
                        help='path to segmentation model weights')
    parser.add_argument('--seg_config_path', type=str, default="/home/clc/公共的/wyf/DPEC-VM-main/FR/config.yaml",
                        help='path to segmentation model config')
    parser.add_argument('--seg_loss_weight', type=float, default=1.0,
                        help='weight for segmentation loss')
    parser.add_argument('--use_gt_seg', type=bool, default=True,
                        help='use ground truth segmentation masks instead of predicted ones')

    # 移除命令行参数，改为在配置中直接设置
    opt = parser.parse_args()

    # 在配置中直接设置checkpoint路径
    opt.resume_checkpoint = ''  # 设置为空字符串表示不恢复，如果要恢复训练，请设置具体的checkpoint路径
    # 例如：opt.resume_checkpoint = 'models/VMLL_LOLv1_pre/last.pth'

    return opt


def checkpoint(model, optimizer, epoch, best_psnr, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_out_path = os.path.join(save_folder, "best.pth")

    # 保存更多训练状态信息
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr
    }
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def checkpoint_last(model, optimizer, epoch, best_psnr, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_out_path = os.path.join(save_folder, "last.pth")

    # 保存更多训练状态信息
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr
    }
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # 修改 load_checkpoint 函数中的加载方式
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best PSNR: {best_psnr}")
        return start_epoch, best_psnr
    else:
        print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        return 0, 0


def log_metrics(logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


best_psnr = 0

import shutil
import os


def copy_files_to_destination(file_list, destination_folder):
    for file_path in file_list:
        try:
            # 构建目标文件路径
            destination_path = os.path.join(destination_folder, os.path.basename(file_path))

            # 复制文件
            shutil.copy2(file_path, destination_path)
        except Exception as e:
            print(f"错误: {file_path} - {str(e)}")


def delete_files(file_list):
    for file_path in file_list:
        try:
            # 删除文件
            os.remove(file_path)
        except OSError as e:
            print(f"错误: {file_path} - {e.strerror}")


def eval(model, optimizer, epoch, writer, txt_write, opt, best_psnr, seg_model=None):
    print("==> Start testing")
    device = torch.device('cuda:' + opt.deviceid)
    tStart = time.time()
    trans = transforms.Compose([
        ToTensor()
    ])
    channel_swap = (1, 2, 0)
    model.eval()
    # Pay attention to the data structure
    test_LL_folder = os.path.join(opt.testset, "low")
    test_NL_folder = os.path.join(opt.testset, "high")
    test_est_folder = os.path.join(opt.output, opt.modelname, 'last')
    test_est_folder_best = os.path.join(opt.output, opt.modelname, 'best')

    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    try:
        os.stat(test_est_folder_best)
    except:
        os.makedirs(test_est_folder_best)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            LL_images = Image.open(test_LL_list[i]).convert('RGB')
            width, height = LL_images.size

            # 计算新的宽度和高度，使其都是8的倍数
            new_width = width - (width % 8)
            new_height = height - (height % 8)

            # 调整图像大小
            resized_LL_image = LL_images.resize((new_width, new_height))
            img_in = trans(resized_LL_image)

            LL_tensor = img_in.unsqueeze(0).to(device)

            # 使用模型进行推理
            prediction = model(LL_tensor)
            # 如果需要，进行预测结果的后处理
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

            # 将NumPy数组转换回Pillow图像
            est_image = Image.fromarray(prediction)

            # 将调整后的预测结果图像调整回原始尺寸
            resized_est_image = est_image.resize((width, height), resample=Image.BILINEAR)

            # 保存调整后的预测结果图像
            resized_est_image.save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, channel_axis=-1)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    save_folder = os.path.join('models', opt.modelname)

    if best_psnr > psnr_score:
        delete_files(est_list)
        file_checkpoint_last = checkpoint_last(model, optimizer, epoch, best_psnr, save_folder)
    else:
        file_checkpoint = checkpoint(model, optimizer, epoch, psnr_score, save_folder)
        best_psnr = psnr_score
        copy_files_to_destination(est_list, test_est_folder_best)
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    print("best_psnr:", best_psnr)
    writer.add_scalar('psnr', psnr_score, epoch)
    writer.add_scalar('ssim', ssim_score, epoch)
    txt_write.write(
        'EPOCH: ' + str(epoch) + ',' + ' PSNR ' + str(psnr_score)[:6] + ',' + ' SSIM ' + str(ssim_score)[:5] + '\n')
    return psnr_score, ssim_score, best_psnr


def logging(dircname):
    if not os.path.exists(dircname):
        os.makedirs(dircname)
    writer = SummaryWriter(dircname, opt.modelname)

    return writer


def C_DIVIDE_A(tensor1, tesnor2):
    C = tensor1.cpu()
    A = tesnor2.cpu()
    c_np = C.detach().numpy()
    a_np = A.detach().numpy()
    img_ret1 = cv2.divide(a_np, c_np)
    img_ret2 = cv2.divide(c_np, a_np)
    device = torch.device('cuda:' + opt.deviceid)
    return torch.from_numpy(img_ret1).to(device), torch.from_numpy(img_ret2).to(device)


def main(opt):
    writer = logging(os.path.join('tensorboard', opt.modelname))
    txt_name = 'metrics_' + opt.modelname + '.txt'

    # 检查是否从checkpoint恢复
    resume_training = bool(opt.resume_checkpoint and opt.resume_checkpoint != '')

    # 以追加模式打开文件，这样恢复训练时不会覆盖之前的记录
    if resume_training and os.path.exists(txt_name):
        txt_write = open(txt_name, mode='a')
        txt_write.write('\n' + '=' * 50 + '\n')
        txt_write.write('Resuming training from checkpoint: ' + opt.resume_checkpoint + '\n')
        txt_write.write('Resuming at: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        txt_write.write('=' * 50 + '\n')
    else:
        txt_write = open(txt_name, mode='w')
        if resume_training:
            txt_write.write('Starting training from checkpoint: ' + opt.resume_checkpoint + '\n')

    with txt_write:
        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        setup_seed(opt.seed)
        if cuda:
            cudnn.benchmark = True
        gpus_list = range(opt.gpus)

        # =============================#
        #   Prepare training data      #
        # =============================#
        print('===> Prepare training data')
        print('#### Now dataset is LOL ####')

        # 使用新的数据集类，加载分割标签
        train_set = EnhancedDataset(opt.trainset, augment=True, patch_size=opt.patch_size)
        training_data_loader = DataLoader(dataset=train_set, num_workers=30, batch_size=opt.batchSize,
                                          pin_memory=True, shuffle=True, drop_last=True)

        # =============================#
        #          Build model         #
        # =============================#
        print('===> Build model')
        lighten = net()
        device = torch.device('cuda:' + opt.deviceid)
        lighten.to(device)
        print('---------- Networks architecture -------------')
        print_network(lighten)
        print('----------------------------------------------')

        # =============================#
        #     Load segmentation model  #
        # =============================#
        print('===> Load segmentation model')
        seg_model = None
        seg_loss_fn = SegmentationLoss()
        if os.path.exists(opt.seg_weight_path):
            seg_model = load_segmentation_model(opt.seg_weight_path, opt.seg_config_path)
            seg_model.to(device)
            print('Segmentation model loaded successfully')
        else:
            print(f'Warning: Segmentation model weights not found at {opt.seg_weight_path}')
            print('Training without segmentation loss')

        # =============================#
        #         Loss function        #
        # =============================#
        L1_criterion = nn.L1Loss()
        TV_loss = TVLoss()
        mse_loss = torch.nn.MSELoss()
        ssim = pytorch_ssim.SSIM()
        percep_loss = VGGPerceptualLoss()
        smooth_criterion = nn.SmoothL1Loss()
        if cuda:
            gpus_list = range(opt.gpus)
            mse_loss = mse_loss.to(device)
            L1_criterion = L1_criterion.to(device)
            TV_loss = TV_loss.to(device)
            ssim = ssim.to(device)
            percep_loss = percep_loss.to(device)
            smooth_criterion = smooth_criterion.to(device)
            seg_loss_fn = seg_loss_fn.to(device)

        # =============================#
        #         Optimizer            #
        # =============================#
        parameters = [lighten.parameters()]
        # 使用 Adam 优化器
        optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

        # =============================#
        #      Load checkpoint         #
        # =============================#
        save_folder = os.path.join('models', opt.modelname)
        start_epoch = opt.start_iter
        global best_psnr

        # 如果配置中指定了checkpoint路径，则从该checkpoint恢复训练
        if resume_training:
            print(f"Resuming training from checkpoint: {opt.resume_checkpoint}")
            start_epoch, best_psnr = load_checkpoint(lighten, optimizer, opt.resume_checkpoint)
        else:
            print("Starting training from scratch")
            best_psnr = 0

        # =============================#
        #         Training             #
        # =============================#

        for epoch in range(start_epoch, opt.nEpochs + 1):
            print('===> training epoch %d' % epoch)
            epoch_loss = 0
            lighten.train()

            tStart_epoch = time.time()
            for iteration, batch in enumerate(training_data_loader, 1):
                over_Iter = epoch * len(training_data_loader) + iteration
                optimizer.zero_grad()
                LL_t, NL_t, seg_gt = batch[0], batch[1], batch[2]  # 现在batch包含三个元素
                if cuda:
                    LL_t = LL_t.to(device)
                    NL_t = NL_t.to(device)
                    seg_gt = seg_gt.to(device)
                t0 = time.time()
                LL_t_flatten = torch.flatten(LL_t)
                pred_t = lighten(LL_t)
                ori, ori_1 = C_DIVIDE_A(NL_t, LL_t)
                label, label_1 = C_DIVIDE_A(pred_t, LL_t)

                his_loss = histogram_loss(pred_t, NL_t)
                histogram_loss_B = histogram_loss(ori, label)

                pred_t_flatten = torch.flatten(pred_t)

                inner_loss = torch.dot(pred_t_flatten, LL_t_flatten) / (LL_t.shape[0]) / (LL_t.shape[1]) / (
                    LL_t.shape[2]) / (LL_t.shape[3])
                ssim_loss = 1 - ssim(pred_t, NL_t)
                tv_loss = TV_loss(pred_t)
                p_loss = percep_loss(pred_t, NL_t)
                smoothloss = smooth_criterion(pred_t, NL_t)

                # 计算分割损失
                seg_loss = 0
                iou_loss = 0
                dice_loss = 0
                if opt.use_gt_seg:
                    # 使用真实分割标签计算损失
                    if seg_model is not None:
                        # 为增强后的图像生成分割掩码
                        pred_seg = generate_segmentation_mask(seg_model, pred_t)
                        # 计算分割损失（与真实分割标签比较）
                        iou_loss, dice_loss = seg_loss_fn(pred_seg, seg_gt)
                        seg_loss = iou_loss + dice_loss
                else:
                    # 使用分割模型预测的分割结果计算损失
                    if seg_model is not None:
                        # 为增强后的图像生成分割掩码
                        pred_seg = generate_segmentation_mask(seg_model, pred_t)
                        # 为真实图像生成分割掩码
                        gt_seg = generate_segmentation_mask(seg_model, NL_t)
                        # 计算分割损失
                        iou_loss, dice_loss = seg_loss_fn(pred_seg, gt_seg)
                        seg_loss = iou_loss + dice_loss

                # 总损失函数，包含分割损失
                loss = (2 * ssim_loss + 1.2 * p_loss + 0.8 * smoothloss +
                        1 * inner_loss + 0.01 * tv_loss + 1 * his_loss +
                        1 * histogram_loss_B + 0.1 * seg_loss)

                writer.add_scalar('ssim_loss', ssim_loss, over_Iter)
                writer.add_scalar('tv_loss', tv_loss * 0.01, over_Iter)
                writer.add_scalar('perceptual_loss', p_loss, over_Iter)
                writer.add_scalar('smooth_loss', smoothloss, over_Iter)
                writer.add_scalar('inner_loss', inner_loss, over_Iter)
                writer.add_scalar('histogram_loss_B', histogram_loss_B, over_Iter)
                if seg_model is not None:
                    writer.add_scalar('segmentation_loss', seg_loss, over_Iter)
                    writer.add_scalar('iou_loss', iou_loss, over_Iter)
                    writer.add_scalar('dice_loss', dice_loss, over_Iter)

                loss.backward()
                optimizer.step()
                t1 = time.time()
                epoch_loss += loss
                if iteration % 10 == 0:
                    print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                          end=" ==> ")
                    logs = {
                        "loss": loss.data,
                        "ssim_loss": ssim_loss.data,
                        "tv_loss": tv_loss.data,
                        "percep_loss": p_loss.data,
                        "smooth_loss": smoothloss,
                        "inner_loss": inner_loss,
                        "histogram_loss_B": histogram_loss_B,
                    }
                    if seg_model is not None:
                        logs["seg_loss"] = seg_loss.data
                        logs["iou_loss"] = iou_loss.data
                        logs["dice_loss"] = dice_loss.data

                    log_metrics(logs, over_Iter)
                    print("time: {:.4f} s".format(t1 - t0))

            writer.add_scalar('epoch_loss', float(epoch_loss / len(training_data_loader)), epoch)
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
                training_data_loader), time.time() - tStart_epoch))
            if epoch % (opt.snapshots) == 0:
                psnr_score, ssim_score, best_psnr = eval(lighten, optimizer, epoch, writer, txt_write, opt, best_psnr,
                                                         seg_model)
                logs = {
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                }
                log_metrics(logs, epoch, end_str="\n")

            # 使用余弦退火更新学习率，并设置最小学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_min + (opt.lr - opt.lr_min) * (
                        1 + np.cos(np.pi * epoch / opt.nEpochs)) / 2
                print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        print("======>>>Finished time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # 训练完成后保存最终的checkpoint
        save_folder = os.path.join('models', opt.modelname)
        checkpoint_last(lighten, optimizer, opt.nEpochs, best_psnr, save_folder)


if __name__ == '__main__':
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    opt = cfg()

    # 在这里设置要恢复的checkpoint路径
    # 如果要恢复训练，取消下面一行的注释并设置正确的路径
    #opt.resume_checkpoint = 'models/VMLL_LOLv1_pre/last.pth'

    main(opt)