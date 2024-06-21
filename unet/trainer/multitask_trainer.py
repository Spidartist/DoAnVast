import wandb
from dataset.Multitask import MultiTask
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss
from score.score import DiceScore, IoUScore, MicroMacroDiceIoU
from model.vit import vit_base, vit_huge
from model.unetr import UNETR
from utils.lr import get_warmup_cosine_lr
from utils.helper import load_state_dict_wo_module, AverageMeter, ScoreAverageMeter, GetItem, GetItemBinary
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, json_path,
            root_path, wandb_token,
            num_freeze=10, max_lr=1e-3, img_size=256
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.json_path = json_path
        self.root_path = root_path
        self.num_freeze= num_freeze
        self.wandb_token = wandb_token
        self.BASE_LR = 1e-6
        self.MAX_LR = max_lr
        self.img_size = (img_size, img_size)
        if self.type_pretrained == "endoscopy" or self.type_pretrained == "endoscopy1" or self.type_pretrained == "none" or self.type_pretrained == "endoscopy2" or self.type_pretrained == "endoscopy3":
            self.batch_size = 8  # old = 16
        elif self.type_pretrained == "im1k":
            self.batch_size = 1
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
        self.cls_loss = WeightedPosCELoss().to(self.device)
        self.bi_cls_loss = WeightedBCELoss().to(self.device)

    def init_score(self):
        self.dice_score = DiceScore().to(self.device)
        self.iou_score = IoUScore().to(self.device)
        self.dice_IoU = MicroMacroDiceIoU().to(self.device)

    def init_model(self):
        if self.type_pretrained == "endoscopy":
            encoder = vit_base(img_size=[256])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
            print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        elif self.type_pretrained == "endoscopy1":
            encoder = vit_base(img_size=[256])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300_17_5_crop.pth.tar")
            print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300_17_5_crop.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        elif self.type_pretrained == "endoscopy2":
            encoder = vit_base(img_size=[256])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
            print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        elif self.type_pretrained == "endoscopy3":
            encoder = vit_base(img_size=[256])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
            print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        elif self.type_pretrained == "none":
            encoder = vit_base(img_size=[256])
            print(self.type_pretrained)
        elif self.type_pretrained == "im1k":
            encoder = vit_huge(img_size=[448])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/IN1K-vit.h.16-448px-300e.pth.tar")
            new_state_dict = load_state_dict_wo_module(ckpt["target_encoder"])
            encoder.load_state_dict(new_state_dict)
        self.net = UNETR(img_size=self.img_size[0], backbone="ijepa", encoder=encoder, task="multitask")
        self.net.to(self.device)
        if self.num_freeze > 0:
            self.net.freeze_encoder()

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
            project="Multitask",
            name=f"freeze:{self.num_freeze}-max_lr:{self.MAX_LR}-img_size:{self.img_size}",
            config={
                "batch": self.batch_size,
                "MAX_LR": self.MAX_LR,
                "BASE_LR": self.BASE_LR,
                "img_size": self.img_size,
                "epoch_num": self.epoch_num
            },
        )

    def init_data_loader(self):
        train_dataset = MultiTask(root_path=self.root_path, mode="train", path=self.json_path, img_size=self.img_size[0])
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        valid_dataset = MultiTask(root_path=self.root_path, mode="test", path=self.json_path, img_size=self.img_size[0])
        self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def run(self):
        for epoch in range(self.epoch_num):            
            train_epoch_loss, train_epoch_pos_loss, train_epoch_dmg_loss, train_epoch_seg_loss, \
        train_epoch_pos_acc, train_epoch_dmg_acc, train_epoch_hp_acc = self.train_one_epoch()
            wandb.log(
                    {
                        "train_epoch_loss": train_epoch_loss,
                        "train_epoch_pos_loss": train_epoch_pos_loss,
                        "train_epoch_dmg_loss": train_epoch_dmg_loss,
                        "train_epoch_seg_loss": train_epoch_seg_loss,
                        "train_epoch_pos_acc": train_epoch_pos_acc,
                        "train_epoch_dmg_acc": train_epoch_dmg_acc,
                        "train_epoch_hp_acc": train_epoch_hp_acc,
                    },
                    step=epoch 
                )
            valid_epoch_loss, val_epoch_pos_loss, val_epoch_dmg_loss, val_epoch_seg_loss, \
        val_epoch_pos_acc, val_epoch_dmg_acc, val_epoch_hp_acc, micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image = self.valid_one_epoch()

            wandb.log(
                {
                    "valid_loss": valid_epoch_loss,
                    "val_epoch_pos_loss": val_epoch_pos_loss,
                    "val_epoch_dmg_loss": val_epoch_dmg_loss,
                    "val_epoch_seg_loss": val_epoch_seg_loss,
                    "val_epoch_pos_acc": val_epoch_pos_acc,
                    "val_epoch_dmg_acc": val_epoch_dmg_acc,
                    "val_epoch_hp_acc": val_epoch_hp_acc,
                    "valid_micro_dice_score": micro_dice_score,
                    "valid_micro_iou_score": micro_iou_score,
                    "valid_macro_dice_score": macro_dice_score,
                    "valid_macro_iou_score": macro_iou_score,
                    "valid_image_visualize": vis_image
                },
                step=epoch
            )
               
            
            

    def train_one_epoch(self):
        steps_per_epoch = len(self.train_data_loader)
        total_steps = steps_per_epoch * self.epoch_num
        self.net.train()
        epoch_loss = AverageMeter()
        epoch_pos_loss = 0
        epoch_dmg_loss = 0
        epoch_seg_loss = 0
        epoch_hp_loss = 0

        # total_pos: total number of records that have position label
        total_pos_correct = 0
        total_pos = 0

        # total_dmg: total number of records that have damage label
        total_dmg_correct = 0
        total_dmg = 0

        # total_hp: total number of records that have hp label
        total_hp_correct = 0
        total_hp = 0

        tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
        for data in tk0:
            img, mask, position_label, damage_label, segment_weight, hp_label = data

            num_records_have_pos = (position_label != -1).sum().item()
            total_pos += num_records_have_pos

            num_records_have_dmg = (damage_label != -1).sum().item()
            total_dmg += num_records_have_dmg

            num_records_have_hp = (hp_label != -1).sum().item()
            total_hp += num_records_have_hp

            img = img.float().to(self.device)
            mask = mask.float().to(self.device)
            position_label = position_label.to(self.device)
            damage_label = damage_label.to(self.device)
            segment_weight = segment_weight.to(self.device)
            hp_label = hp_label.float().to(self.device)

            lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
            self.optimizer.param_groups[0]['lr'] = 0.1 * lr
            self.optimizer.param_groups[1]['lr'] = lr

            pos_out, dmg_out, hp_out, seg_out = self.net(img)

            loss1 = self.cls_loss(pos_out, position_label)
            loss2 = self.cls_loss(dmg_out, damage_label)
            loss3 = self.seg_loss(seg_out, mask, segment_weight)
            loss4 = self.bi_cls_loss(hp_out, hp_label)

            loss = loss1 + loss2 + loss3 + loss4

            epoch_loss.update(loss)
            epoch_pos_loss += loss1
            epoch_dmg_loss += loss2
            epoch_seg_loss += loss3
            epoch_hp_loss += loss4

            total_pos_correct += self.get_item(pos_out, position_label)
            total_dmg_correct += self.get_item(dmg_out, damage_label)
            total_hp_correct += self.get_item_binary(hp_out, hp_label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
        epoch_pos_acc = np.nan if total_pos == 0 else total_pos_correct/total_pos
        epoch_dmg_acc = np.nan if total_dmg == 0 else total_dmg_correct/total_dmg 
        epoch_hp_acc = np.nan if total_hp == 0 else total_hp_correct/total_hp


        return epoch_loss.avg, epoch_pos_loss, epoch_dmg_loss, epoch_seg_loss, \
        epoch_pos_acc, epoch_dmg_acc, epoch_hp_acc

    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()

        epoch_loss = AverageMeter()
        epoch_pos_loss = 0
        epoch_dmg_loss = 0
        epoch_seg_loss = 0
        epoch_hp_loss = 0
        vis_image = None
        epoch_dice_score = ScoreAverageMeter(self.device)
        epoch_iou_score = ScoreAverageMeter(self.device)
        epoch_intersection = ScoreAverageMeter(self.device)
        epoch_union = ScoreAverageMeter(self.device)
        epoch_intersection2 = ScoreAverageMeter(self.device)
        epoch_total_area = ScoreAverageMeter(self.device)

        # total_pos: total number of records that have position label
        total_pos_correct = 0
        total_pos = 0

        # total_dmg: total number of records that have damage label
        total_dmg_correct = 0
        total_dmg = 0

        # total_hp: total number of records that have hp label
        total_hp_correct = 0
        total_hp = 0

        tk0 = tqdm(self.valid_data_loader, total=steps_per_epoch)
        with torch.no_grad():
            for data in tk0:
                img, mask, position_label, damage_label, segment_weight, hp_label = data

                num_records_have_pos = (position_label != -1).sum().item()
                total_pos += num_records_have_pos

                num_records_have_dmg = (damage_label != -1).sum().item()
                total_dmg += num_records_have_dmg

                num_records_have_hp = (hp_label != -1).sum().item()
                total_hp += num_records_have_hp

                img = img.float().to(self.device)
                mask = mask.float().to(self.device)
                position_label = position_label.to(self.device)
                damage_label = damage_label.to(self.device)
                segment_weight = segment_weight.to(self.device)
                hp_label = hp_label.float().to(self.device)

                pos_out, dmg_out, hp_out, seg_out = self.net(img)

                loss1 = self.cls_loss(pos_out, position_label)
                loss2 = self.cls_loss(dmg_out, damage_label)
                loss3 = self.seg_loss(seg_out, mask, segment_weight)
                loss4 = self.bi_cls_loss(hp_out, hp_label)

                loss = loss1 + loss2 + loss3 + loss4

                epoch_loss.update(loss)
                epoch_pos_loss += loss1
                epoch_dmg_loss += loss2
                epoch_seg_loss += loss3
                epoch_hp_loss += loss4

                iou, dice, intersection, union, intersection2, total_area = self.dice_IoU(seg_out, mask)
                epoch_iou_score.update(iou.to(self.device))
                epoch_dice_score.update(dice.to(self.device))
                epoch_intersection.update(intersection.to(self.device))
                epoch_union.update(union.to(self.device))
                epoch_intersection2.update(intersection2.to(self.device))
                epoch_total_area.update(total_area.to(self.device))

                total_pos_correct += self.get_item(pos_out, position_label)
                total_dmg_correct += self.get_item(dmg_out, damage_label)
                total_hp_correct += self.get_item_binary(hp_out, hp_label)



            seg_img = seg_out[0]
            seg_img[seg_img <= 0.5] = 0
            seg_img[seg_img > 0.5] = 1

            images = torch.cat([seg_img, mask[0]], dim=1)
            images = wandb.Image(images)
            vis_image = images
        
        micro_iou_score = epoch_intersection.lst_tensor.sum()/epoch_union.lst_tensor.sum()
        micro_dice_score = epoch_intersection2.lst_tensor.sum()/epoch_total_area.lst_tensor.sum()
        macro_iou_score = epoch_iou_score.lst_tensor.mean()
        macro_dice_score = epoch_dice_score.lst_tensor.mean()

        epoch_pos_acc = np.nan if total_pos == 0 else total_pos_correct/total_pos
        epoch_dmg_acc = np.nan if total_dmg == 0 else total_dmg_correct/total_dmg 
        epoch_hp_acc = np.nan if total_hp == 0 else total_hp_correct/total_hp

        return epoch_loss.avg, epoch_pos_loss, epoch_dmg_loss, epoch_seg_loss, \
        epoch_pos_acc, epoch_dmg_acc, epoch_hp_acc, \
        micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image


