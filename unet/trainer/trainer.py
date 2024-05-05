import wandb
from dataset.TonThuong import TonThuong
from loss.loss import DiceBCELoss
from score.score import DiceScore
from model.vit import vit_base
from model.unetr import UNETR
from utils.lr import get_warmup_cosine_lr

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, type_damaged, json_path,
            root_path, wandb_token
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.type_damaged = type_damaged
        self.json_path = json_path
        self.root_path = root_path
        self.wandb_token = wandb_token
        self.batch_size = 16
        self.BASE_LR = 1e-6
        self.MAX_LR = 1e-3
        self.img_size = (256, 256)
        self.epoch_num = 100
        self.save_freq = 1
        self.save_path = "/logs/"
        self.warmup_epochs = 2
        self.global_step = 0

        self.init_logger()
        self.init_data_loader()
        self.init_loss()
        self.init_score()
        self.init_model()
        self.init_optim()
        self.display_info()

    def display_info(self):
        print('epoch            : %d' % self.epoch_num)
        print('batch_size       : %d' % self.batch_size)
        print('save_freq        : %d' % self.save_freq)
        print('img_size         : (%d, %d)' % (self.img_size[0], self.img_size[1]))
        print('BASE_LR          : %s' % self.BASE_LR)
        print('MAX_LR           : %s' % self.MAX_LR)
        print('warmup_epochs:   : %d' % self.warmup_epochs)
        print('device           : %s' % self.device)
        print('log dir          : %s' % self.save_path)
        print('model has {} parameters in total'.format(sum(x.numel() for x in self.net.parameters())))

    def init_loss(self):
        self.seg_loss = DiceBCELoss().to(self.device)

    def init_score(self):
        self.dice_score = DiceScore().to(self.device)


    def init_model(self):
        encoder = vit_base(img_size=[256])
        if self.type_pretrained == "endoscopy":
            print(torch.cuda.is_available())
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        else:
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])

        self.net = UNETR(img_size=256, backbone="ijepa", encoder=encoder)
        self.net.to(self.device)

    def init_optim(self):
        base, head = [], []
        for name, param in self.net.named_parameters():
            if 'encoder' in name:
                base.append(param)
            else:
                head.append(param)

        self.optimizer = optim.Adam([{'params': base}, {'params': head}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    def init_logger(self):
        wandb.login(key=self.wandb_token)
        wandb.init(
            project="TonThuong",
            name=f"{self.type_damaged}-{self.type_pretrained}",
            config={
                "batch": self.batch_size,
                "MAX_LR": self.MAX_LR,
                "BASE_LR": self.BASE_LR,
                "img_size": self.img_size,
                "epoch_num": self.epoch_num
            },
        )

    def init_data_loader(self):
        train_dataset = TonThuong(root_path=self.root_path, mode="train", type=self.type_damaged)
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        valid_dataset = TonThuong(root_path=self.root_path, mode="test", type=self.type_damaged)
        self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)

    def run(self):
        for _ in range(self.epoch_num):
            train_epoch_loss, train_epoch_score = self.train_one_epoch()
            valid_epoch_loss, valid_epoch_score, vis_image = self.valid_one_epoch()

            wandb.log(
                {
                    "train_epoch_loss": train_epoch_loss,
                    "train_epoch_score": train_epoch_score,
                    "valid_epoch_loss": valid_epoch_loss,
                    "valid_epoch_score": valid_epoch_score,
                    "valid_image_visualize": vis_image
                }
            )

    def train_one_epoch(self):
        steps_per_epoch = len(self.train_data_loader)
        total_steps = steps_per_epoch * self.epoch_num
        self.net.train()
        epoch_loss = 0
        epoch_dice_score = 0

        tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
        for data in tk0:
            img, mask = data

            img = img.float().to(self.device)
            mask = mask.float().to(self.device)

            lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
            self.optimizer.param_groups[0]['lr'] = 0.1 * lr
            self.optimizer.param_groups[1]['lr'] = lr

            seg_out = self.net(img)

            loss3 = self.seg_loss(seg_out, mask, 1)

            epoch_loss += loss3.item()

            self.optimizer.zero_grad()
            loss3.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.net.eval()
                # score = dice_score(seg_out, mask[1], segment_weight)
                score = self.dice_score(seg_out, mask, 1)
                epoch_dice_score += score.item()

            # if global_step % self.save_freq == 0 or global_step == total_steps-1:
            #     torch.save(self.net.state_dict(), self.save_path + f'/model-{self.type_pretrained}-{self.type_damaged}-best.pt')

            self.global_step += 1

        return epoch_loss/steps_per_epoch, epoch_dice_score/steps_per_epoch


    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()
        epoch_loss = 0
        epoch_dice = 0

        vis_image = None

        tk0 = tqdm(self.valid_data_loader, total=steps_per_epoch)
        with torch.no_grad():
            for data in tk0:
                img, mask = data

                img = img.float().to(self.device)
                mask = mask.float().to(self.device)

                seg_out = self.net(img)

                loss3 = self.seg_loss(seg_out, mask, 1)

                epoch_loss += loss3.item()

                score = self.dice_score(seg_out, mask, 1)

                epoch_dice += score.item()

                seg_img = seg_out[0]
                seg_img[seg_img <= 0.5] = 0
                seg_img[seg_img > 0.5] = 1

                images = torch.cat([seg_img, mask[0]], dim=1)
                images = wandb.Image(images)
                vis_image = images

        # if epoch_loss < best_epoch_loss:
        #     print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        #     best_epoch_loss = epoch_loss
        #     torch.save(self.net.state_dict(), self.save_path + f'/model-{type}-{loai_ton_thuong}-best.pt')
        return epoch_loss/steps_per_epoch , epoch_dice/steps_per_epoch, vis_image

    
