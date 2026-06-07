import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip

class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):
        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split * len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split * len(self.img_file)):]
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            # Fix_RandomRotation(),  # 确保自定义旋转正确实现
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        file_name = img_file  # 保存原始文件名

        with open(os.path.join(self.data_path, img_file), 'rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        with open(os.path.join(self.data_path, gt_file), 'rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt, file_name  # 返回图片名字

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
        return img_list

    def __len__(self):
        return len(self.img_file)