from model.unetr import UNETR
from model.vit import vit_base
import torch

encoder = vit_base(img_size=[256])
ckpt = torch.load("/mnt/quanhd/ijepa/logs/jepa-ep300.pth.tar")
encoder.load_state_dict(ckpt["target_encoder"])

x = torch.ones((1, 3, 256, 256))

unetr = UNETR(img_size=256, backbone="ijepa", encoder=encoder)
base, head = [], []
tmp = []
for name, param in unetr.named_parameters():
    if 'encoder' in name:
        base.append(param)
        tmp.append(name)
    else:
        head.append(param)
print(tmp)