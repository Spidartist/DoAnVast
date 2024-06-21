from collections import OrderedDict
import torch
import torch.nn as nn
from score.score import MicroMacroDiceIoUMultitask

def load_state_dict_wo_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

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