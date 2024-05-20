from collections import OrderedDict
import torch
import torch.nn as nn

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