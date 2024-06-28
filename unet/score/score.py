import torch
import torch.nn as nn
import numpy as np

class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, seg_weight: torch.Tensor, smooth=1e-15):
        # print(inputs.size(), targets.size())
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]

        inputs = torch.sigmoid(torch.flatten(inputs))
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1
        targets = torch.flatten(targets.float())
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return torch.nansum(dice)
    
class MicroMacroDiceIoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MicroMacroDiceIoU, self).__init__()
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1

        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = inputs.sum(dim=(1, 2, 3)) - tp
        fn = targets.sum(dim=(1, 2, 3)) - tp
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        intersection = tp
        union = tp + fp + fn
        intersection2 = 2*tp
        total_area = 2 * tp + fp + fn
        return iou, dice, intersection, union, intersection2, total_area
    
class MicroMacroDiceIoUMultitask(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MicroMacroDiceIoUMultitask, self).__init__()
    def forward(self, inputs, targets, seg_weight: torch.Tensor, dmg_label, batch_dmg_labels: torch.Tensor, smooth=1e-6):
        # print(seg_weight)
        # print(batch_dmg_labels)
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]
        batch_dmg_labels = batch_dmg_labels[seg_weight != 0]
        inputs = inputs[batch_dmg_labels == dmg_label]
        targets = targets[batch_dmg_labels == dmg_label]
        if inputs.shape[0] == 0:
            return False, [0, 0, 0, 0, 0, 0]

        inputs = torch.sigmoid(inputs)
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1

        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = inputs.sum(dim=(1, 2, 3)) - tp
        fn = targets.sum(dim=(1, 2, 3)) - tp
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        intersection = tp + smooth
        union = tp + fp + fn + smooth
        intersection2 = 2*tp + smooth
        total_area = 2 * tp + fp + fn + smooth
        return True, [iou, dice, intersection, union, intersection2, total_area]
    
class FbetaScore(nn.Module):
    def __init__(self, weight=None, size_average=True, beta=1):
        super(FbetaScore, self).__init__()

    def precision(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

    def recall(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        return (intersection + 1e-15) / (y_true.sum() + 1e-15)

    def Fbeta(y_true, y_pred, beta=1):
        p = precision(y_true,y_pred)
        r = recall(y_true, y_pred)
        return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)
    
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1

        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = inputs.sum(dim=(1, 2, 3)) - tp
        fn = targets.sum(dim=(1, 2, 3)) - tp
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        intersection = tp
        union = tp + fp + fn
        intersection2 = 2*tp
        total_area = 2 * tp + fp + fn
        return iou, dice, intersection, union, intersection2, total_area
    
class IoUScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUScore, self).__init__()

    def forward(self, inputs, targets, seg_weight: torch.Tensor, smooth=1e-15):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = torch.flatten(inputs)
        # targets = torch.flatten(targets.float())
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]

        inputs = torch.sigmoid(torch.flatten(inputs))
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1
        targets = torch.flatten(targets.float())
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return torch.nansum(IoU)

# class IoUScore(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(IoUScore, self).__init__()

#     def forward(self, inputs, targets, smooth=1e-15):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = torch.flatten(inputs)
#         targets = torch.flatten(targets.float())
        
#         #intersection is equivalent to True Positive count
#         #union is the mutually inclusive area of all labels & predictions 
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection 
        
#         IoU = (intersection + smooth)/(union + smooth)
                
#         return IoU


class MultiClassesDiceScore(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_score(self, input, target, classes, ignore_index=-100, smooth=1e-15):
        """ Functional dice score calculation on multiple classes. """

        target = target.long().unsqueeze(1)

        # getting mask for valid pixels, then converting "void class" to background
        valid = target != ignore_index
        target[target == ignore_index] = 0
        valid = valid.float()

        # converting to onehot image with class channels
        onehot_target = torch.LongTensor(target.shape[0], classes, target.shape[-2], target.shape[-1]).zero_().cuda()
        onehot_target.scatter_(1, target, 1)  # write ones along "channel" dimension
        # classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
        onehot_target = onehot_target.float()

        # keeping the valid pixels only
        onehot_target = onehot_target * valid
        input = input * valid

        dice = 2 * (input * onehot_target).sum([2, 3]) / ((input**2).sum([2, 3]) + (onehot_target**2).sum([2, 3]) + smooth)
        return dice.mean(dim=1)
    
    def forward(self, inputs, targets, classes, seg_weight: torch.Tensor, smooth=1e-12):
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]

        return torch.nansum((1 - self.dice_score(inputs, targets, classes)).mean())
    
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def Fbeta(y_true, y_pred, beta=1):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

class Metric():
    def __init__(self, name):
        self.name = name
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.list_iou_score = []
        self.list_dice_score = []

    def cal(self, predict, mask, smooth=1):
        tp, fp, fn = self.metric(predict, mask)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        iou_score = (tp + smooth) / (tp + fp + fn + smooth)
        dice_score = (2*tp + smooth) / (2*tp + fp + fn + smooth)
        self.list_iou_score.append(iou_score)
        self.list_dice_score.append(dice_score)

    def show(self):
        dice_score = 2 * self.total_tp / \
            (2 * self.total_tp + self.total_fp + self.total_fn)
        iou_score = self.total_tp / \
            (self.total_tp + self.total_fp + self.total_fn)

        print("Evaluate {}".format(self.name))
        print("Dice score micro {}".format(dice_score))
        print("IoU score micro {}".format(iou_score))
        print("Dice score macro {}".format(
            np.array(self.list_dice_score).mean()))
        print("IoU score macro {}".format(np.array(self.list_iou_score).mean()))

    def metric(self, inputs, targets):
        tp = np.sum(inputs * targets)
        fp = np.sum(inputs) - tp
        fn = np.sum(targets) - tp

        return tp, fp, fn