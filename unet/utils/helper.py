from collections import OrderedDict
import torch
import torch.nn as nn
from score.score import MicroMacroDiceIoUMultitask
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import ConfusionMatrix
import io
import cv2


def load_state_dict_wo_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class ConfMatObj:
    def __init__(self, device):
        self.pred = {}
        self.gt = {}
        self.device = device
        self.key_lst = ["pos", "dmg", "hp"]
    def add(self, pos_out, dmg_out, hp_out, pos_label, dmg_label, hp_label):
        for k, out, label in zip(self.key_lst, [pos_out, dmg_out, hp_out], [pos_label, dmg_label, hp_label]):
            if k != "hp":
                preds = out[label != -1]
            else:
                preds = torch.flatten(out[label != -1]).sigmoid()
            gts = label[label != -1]
            if k in self.pred:
                self.pred[k] = torch.cat((self.pred[k], preds))
                self.gt[k] = torch.cat((self.gt[k], gts))
            else:
                self.pred[k] = preds
                self.gt[k] = gts
    def ret_confmat(self, k):
        if k == "pos":
            class_names = ["Hầu họng", "Thực quản", "Tam vị", "Thân vị", "Phình vị", "Hang vị", "Bờ cong lớn", "Bờ cong nhỏ", "Hành tá tràng", "Tá tràng"]
            confmat = ConfusionMatrix(task="multiclass", num_classes=10).to(self.device)
            pred = self.pred[k]
            gts = self.gt[k]
        elif k == "dmg":
            class_names = ["VTGP", "UTTQ", "VTQ", "VLHTT", "UTDD", "VDD/HP", "POLYP"]
            confmat = ConfusionMatrix(task="multiclass", num_classes=7).to(self.device)
            pred = self.pred[k]
            gts = self.gt[k]
        elif k == "hp":
            class_names = ["Lành tính", "Ác tính"]
            confmat = ConfusionMatrix(task="binary", num_classes=2).to(self.device)
            pred = self.pred[k]
            gts = self.gt[k]
        confusion_matrix = confmat(pred, gts)
        conf_img = plot_confusion_matrix(confusion_matrix, class_names, normalize=True)

        return conf_img


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Plots a confusion matrix using matplotlib.

    Args:
    cm (array, shape = [n, n]): confusion matrix
    class_names (list): List of class names
    normalize (bool): Whether to normalize the values to percentages
    """
    cm = cm.cpu().numpy()
    if normalize:
        cm = np.nan_to_num(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7))

    # Big size
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm, cmap='Blues')

    ax.set_title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names, rotation=45)
    ax.set_yticks(tick_marks, class_names)

    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    # To HWC
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

class MicroMacroMeter(object):
    def __init__(self, device, dmg_label):
        self.device = device
        self.dmg_label = dmg_label
        self.epoch_dice_score = ScoreAverageMeter(self.device)
        self.epoch_iou_score = ScoreAverageMeter(self.device)
        self.epoch_intersection = ScoreAverageMeter(self.device)
        self.epoch_union = ScoreAverageMeter(self.device)
        self.epoch_intersection2 = ScoreAverageMeter(self.device)
        self.epoch_total_area = ScoreAverageMeter(self.device)
        self.dice_IoU = MicroMacroDiceIoUMultitask().to(self.device)
    def update(self, seg_out, mask, segment_weight, batch_dmg_labels):
        is_val, [iou, dice, intersection, union, intersection2, total_area] = self.dice_IoU(seg_out, mask, segment_weight, self.dmg_label, batch_dmg_labels)
        if is_val:
            self.epoch_dice_score.update(dice.to(self.device))
            self.epoch_iou_score.update(iou.to(self.device))
            self.epoch_intersection.update(intersection.to(self.device))
            self.epoch_union.update(union.to(self.device))
            self.epoch_intersection2.update(intersection2.to(self.device))
            self.epoch_total_area.update(total_area.to(self.device))
    def ret_val(self):
        micro_iou_score = self.epoch_intersection.lst_tensor.sum()/self.epoch_union.lst_tensor.sum()
        micro_dice_score = self.epoch_intersection2.lst_tensor.sum()/self.epoch_total_area.lst_tensor.sum()
        macro_iou_score = self.epoch_iou_score.lst_tensor.mean()
        macro_dice_score = self.epoch_dice_score.lst_tensor.mean()
        return [micro_iou_score, micro_dice_score, macro_iou_score, macro_dice_score]



class ScoreAverageMeter(object):
    def __init__(self, device):
        self.reset(device)

    def reset(self, device):
        self.val = 0
        self.lst_tensor = torch.tensor([]).to(device)
        self.count = 0

    def update(self, tensor):
        self.val = tensor
        self.lst_tensor = torch.concatenate([self.lst_tensor, tensor])
        self.count += tensor.shape[0]

class GetItem(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        x = torch.softmax(preds, dim=1)
        return x.argmax(dim=1).eq(labels).sum().item()


class GetItemBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, labels):
        x = torch.sigmoid(torch.flatten(preds))
        x[x >= 0.5] = 1
        x[x < 0.5] = 0

        return x.eq(labels).sum().item()