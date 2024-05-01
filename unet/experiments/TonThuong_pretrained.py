from dataset.TonThuong import TonThuong
from model.unetr import UNETR
import torch
from torch.utils.data import DataLoader
from loss.loss import DiceBCELoss
from score.score import DiceScore
from model.vit import vit_base
import torch.optim as optim
import wandb
from utils.lr import get_warmup_cosine_lr
from datetime import datetime

savepath = './log/' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
BASE_LR = 1e-6
MAX_LR = 1e-3
warmup_epochs = 2
batch = 16
epoch_num = 50
save_freq = 200
img_size = (256, 256)

wandb.login(key="cca12c93cb17351580e3f9fd5136347e65a3463d")
wandb.init(
    project="TonThuong",
    config={
        "batch": batch,
        "MAX_LR": MAX_LR,
        "BASE_LR": BASE_LR,
        "img_size": img_size,
        "epoch_num": epoch_num
    },
)

d = TonThuong(root_path="/home/s/DATA/")
data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=True)

if device == 'cuda':
    seg_loss = DiceBCELoss().cuda()
    dice_score = DiceScore().cuda()

encoder = vit_base(img_size=[256])
ckpt = torch.load("/mnt/quanhd/ijepa/logs/jepa-ep300.pth.tar")
encoder.load_state_dict(ckpt["target_encoder"])

net = UNETR(img_size=256, backbone="ijepa", encoder=encoder)
net.train()
if device == 'cuda':
    net.cuda()

base, head = [], []
for name, param in net.named_parameters():
    if 'encoder' in name:
        base.append(param)
    else:
        head.append(param)

optimizer = optim.Adam([{'params': base}, {'params': head}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

print('epoch            : %d' % epoch_num)
print('batch_size       : %d' % batch)
print('save_freq        : %d' % save_freq)
print('img_size         : (%d, %d)' % (img_size[0], img_size[1]))
print('BASE_LR          : %s' % BASE_LR)
print('MAX_LR           : %s' % MAX_LR)
print('warmup_epochs:   : %d' % warmup_epochs)
print('device           : %s' % device)
print('log dir          : %s' % savepath)
print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

steps_per_epoch = len(data_loader)
total_steps = steps_per_epoch * epoch_num
print("Start training...")

global_step = 0
best_epoch_loss = float('inf')

for epoch in range(epoch_num):
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_dice_score = []


    for i, data in enumerate(data_loader):
        img, mask = data

        if device == 'cuda':
            img = img.float().cuda()
            mask = mask.float().cuda()

        lr = get_warmup_cosine_lr(BASE_LR, MAX_LR, global_step, total_steps, steps_per_epoch, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]['lr'] = 0.1 * lr
        optimizer.param_groups[1]['lr'] = lr

        seg_out = net(img)

        loss3 = seg_loss(seg_out, mask, 1)

        epoch_seg_loss += loss3
        epoch_loss += loss3

        optimizer.zero_grad()
        loss3.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            # score = dice_score(seg_out, mask[1], segment_weight)
            score = dice_score(seg_out, mask, 1)
            epoch_dice_score.append(score.item())

        if global_step % save_freq == 0 or global_step == total_steps-1:
            torch.save(net.state_dict(), savepath + '/model-last.pt')

        wandb.log(
            {
                "epoch": epoch + 1,
                "dice_score": score,
                "loss_segment": loss3.item(),
                "lr": lr,
            }
        )

        global_step += 1
    
    if epoch_loss < best_epoch_loss:
        print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        best_epoch_loss = epoch_loss
        torch.save(net.state_dict(), savepath + '/model-best.pt')