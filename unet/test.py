from model.vit import vit_base, vit_huge
from model.unetr import UNETR
import torch
from pprint import pprint
import torch.optim as optim

encoder = vit_base(img_size=[256])
ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar")
print("/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar")
encoder.load_state_dict(ckpt)
print("encoder's state_dict:")
cnt = 0
for param_tensor in encoder.state_dict():
    # print(param_tensor, "\t", encoder.state_dict()[param_tensor])
    cnt += 1
    if cnt == 2:
        break

# pprint(encoder.state_dict())
net = UNETR(img_size=512, backbone="ijepa", encoder=encoder)
# pprint(net)
cnt = 0
for param_tensor in net.state_dict():
    # print(param_tensor, "\t", net.state_dict()[param_tensor])
    cnt += 1
    if cnt == 2:
        break
base, head = [], []
for name, param in net.named_parameters():
    if 'encoder' in name:
        base.append(param)
    else:
        head.append(param)

optimizer = optim.SGD([{'params': base}, {'params': head}], lr=3e-5, weight_decay=0, momentum=0.9)

for idx, group in enumerate(optimizer.param_groups):
    if idx == 1:
        print(group["params"][0])