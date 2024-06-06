from model.vit import vit_huge
import torch
from collections import OrderedDict
import torch.nn as nn
from model.unetr import UNETR
from model.vit import vit_base
from pprint import pprint

encoder = vit_base(img_size=[256])
ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
encoder.load_state_dict(ckpt["target_encoder"])

unetr = UNETR(img_size=256, backbone="ijepa", encoder=encoder)
pprint(unetr.state_dict)

for param in unetr.encoder.parameters():
    print
# encoder = vit_huge(img_size=[448], patch_size=16)


# ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/IN1K-vit.h.16-448px-300e.pth.tar")

# def load_state_dict_wo_module(state_dict):
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     return new_state_dict

# # load params

# encoder.load_state_dict(new_state_dict)

# encoder.load_state_dict(ckpt["target_encoder"])

# x = torch.ones((1, 3, 448, 448))
# proj = nn.Conv2d(3, 1280, kernel_size=16, stride=16)
# print(proj(x).flatten(2).transpose(1, 2).shape)