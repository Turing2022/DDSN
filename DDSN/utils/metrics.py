import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target, threshold=None, predict_b=None):
    # 确保 predict 和 target 是 NumPy 数组
    if torch.is_tensor(predict):
        predict = predict.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()

    # 确保 predict 和 target 是一维的
    predict = predict.flatten()
    target = target.flatten()

    # 确保 predict 是概率值
    predict = np.clip(predict, 0, 1)

    # 确保 target 是二进制值
    target = np.where(target > 0, 1, 0)

    # 计算 AUC
    try:
        auc = roc_auc_score(target, predict)
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
        auc = 0.0

    # 其他指标计算...
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp) if (tp + fp) != 0 else 0
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) != 0 else 0

    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n
