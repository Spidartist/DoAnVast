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
from tqdm import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='have pretrained or from scratch', default="pretrained", type=str)
    parser.add_argument('--tonthuong', help='Ten loai ton thuong', required=True, type=str)
    parser.add_argument('--json-path', help='path to json file', default="/mnt/quanhd/endoscopy/ft_ton_thuong.json", type=str)
    parser.add_argument('--root-path', help='path to root folder of data', default="/home/s/DATA/", type=str)
    args = parser.parse_args()
    return args

def test_one_epoch(
        net, epoch, val_data_loader, device, seg_loss, dice_score,
        loai_ton_thuong, type, savepath, best_epoch_loss
    ):
    net.eval()
    epoch_loss = 0

    tk0 = tqdm(val_data_loader, total=int(len(val_data_loader)))
    with torch.no_grad():
        for data in tk0:
            img, mask = data

            img = img.float().to(device)
            mask = mask.float().to(device)

            seg_out = net(img)

            loss3 = seg_loss(seg_out, mask, 1)

            epoch_loss += loss3

            score = dice_score(seg_out, mask, 1)

            seg_img = seg_out[0]
            seg_img[seg_img <= 0.5] = 0
            seg_img[seg_img > 0.5] = 1

            images = torch.cat([seg_img, mask[0]], dim=1)
            images = wandb.Image(images)

            wandb.log(
                {
                    "test_epoch": epoch + 1,
                    "test_dice_score": score,
                    "test_loss_segment": loss3.item(),
                    "mae_image": images
                }
            )

    if epoch_loss < best_epoch_loss:
        print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        best_epoch_loss = epoch_loss
        torch.save(net.state_dict(), savepath + f'/model-{type}-{loai_ton_thuong}-best.pt')
    return best_epoch_loss




def train_one_epoch(
        net, epoch, train_data_loader, 
        device, optimizer, warmup_epochs, epoch_num,
        seg_loss, dice_score, save_freq, savepath, BASE_LR, MAX_LR, global_step, type, loai_ton_thuong
    ):
    steps_per_epoch = len(train_data_loader)
    total_steps = steps_per_epoch * epoch_num
    net.train()
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_dice_score = []

    tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)))
    for data in tk0:
        img, mask = data

        img = img.float().to(device)
        mask = mask.float().to(device)

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
            torch.save(net.state_dict(), savepath + f'/model-{type}-{loai_ton_thuong}-best.pt')

        wandb.log(
            {
                "train_epoch": epoch + 1,
                "train_dice_score": score,
                "train_loss_segment": loss3.item(),
                "train_lr": lr,
            }
        )

        global_step += 1
    return global_step
    
    
def train_and_val():
    args = parse_args()

    savepath = 'logs/'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    # device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
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
        name=f"{args.tonthuong}-{args.type}",
        config={
            "batch": batch,
            "MAX_LR": MAX_LR,
            "BASE_LR": BASE_LR,
            "img_size": img_size,
            "epoch_num": epoch_num
        },
    )

    train_dataset = TonThuong(root_path="/home/s/DATA/", type=args.tonthuong)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

    val_dataset = TonThuong(root_path="/home/s/DATA/", mode="test", type=args.tonthuong)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=True)

    seg_loss = DiceBCELoss().to(device)
    dice_score = DiceScore().to(device)

    encoder = vit_base(img_size=[256])
    if args.type == "pretrained":
        ckpt = torch.load("/mnt/quanhd/ijepa/logs/jepa-ep300.pth.tar")
        encoder.load_state_dict(ckpt["target_encoder"])

    net = UNETR(img_size=256, backbone="ijepa", encoder=encoder)
    net.train()
    net.to(device)

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

    
    print("Start training...")

    global_step = 0
    best_epoch_loss = float('inf')

    for epoch in range(epoch_num):
        global_step = train_one_epoch(
                        net, epoch, train_data_loader, 
                        device, optimizer, warmup_epochs, 
                        epoch_num, seg_loss, dice_score, save_freq, savepath, BASE_LR, MAX_LR, global_step,
                        args.type, args.tonthuong
                    )
        best_epoch_loss = test_one_epoch(
                            net, epoch, val_data_loader, device, seg_loss, dice_score, args.tonthuong, args.type, savepath, best_epoch_loss
                            )

if __name__ == "__main__":
    train_and_val()