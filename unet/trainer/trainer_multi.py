import wandb
from dataset.Multitask import Multitask
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss
from score.score import DiceScore, IoUScore, MicroMacroDiceIoUMultitask
from model.vit import vit_base, vit_huge
from model.unetr import UNETRMultitask
from model.unet import Unet
# from model.vit_adapter import IJEPAAdapter
from utils.lr import get_warmup_cosine_lr, WarmupCosineSchedule
from utils.helper import load_state_dict_wo_module, AverageMeter, MicroMacroMeter, GetItem, GetItemBinary, ConfMatObj
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, json_path, amp,
            root_path, wandb_token,  min_lr=2e-4, ref_lr=1e-3,
            num_freeze=10, max_lr=1e-6, img_size=256, type_opt="Adam", batch_size=16, accum_iter=16,
            type_encoder="target_encoder", train_ratio=1.0, scale_lr=1, metadata_file="/root/quanhd/DoAn/unet/dataset/data_dir_endounet.json",
            continue_ckpt=""
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.metadata_file = metadata_file
        self.json_path = json_path
        self.root_path = "/root/quanhd/DATA"
        self.continue_ckpt = continue_ckpt
        self.num_freeze= num_freeze
        self.wandb_token = wandb_token
        self.accum_iter = accum_iter
        self.type_encoder = type_encoder
        self.MIN_LR = min_lr
        self.amp = amp
        self.train_ratio = train_ratio
        self.BASE_LR = ref_lr
        self.type_opt = type_opt
        self.MAX_LR = max_lr
        self.img_size = (img_size, img_size)
        self.batch_size = batch_size
        self.epoch_num = 20
        self.save_freq = 1
        self.save_path = "/logs/"
        self.warmup_epochs = 2
        self.global_step = 0
        self.scale_lr = True if scale_lr == 1 else False

        self.init_data_loader()
        self.init_loss()
        self.init_score()
        self.init_model()
        self.init_optim()
        self.init_logger()
        self.display_info()
        if self.continue_ckpt != "":
            self.load_continue_ckpt()
    
    def load_continue_ckpt(self):
        ckpt = torch.load("/root/quanhd/DoAn/unet/snapshots_endoscopy_mae/12.pth")
        self.net.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for i in range(ckpt["scheduler"]):
            print(i)
            self.lr_scheduler.step()

        


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
        if self.amp:
            print("AMP!!!")
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2**14, enabled=self.amp)

    def init_score(self):
        self.dice_score = DiceScore().to(self.device)
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.iou_score = IoUScore().to(self.device)
        self.dice_IoU = MicroMacroDiceIoUMultitask().to(self.device)
        self.get_item = GetItem()
        self.get_item_binary = GetItemBinary()

    def init_model(self):
        if self.type_pretrained == "resnet":
            print("EndoUnet")
            self.net = Unet(classes=1, position_classes=10, damage_classes=7, backbone_name='resnet50', pretrained=True)
            self.net.to(self.device)
        else:
            if self.type_pretrained == "endoscopy":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
                print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "endoscopy1":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300_17_5_crop.pth.tar")
                print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300_17_5_crop.pth.tar")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "endoscopy2":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
                print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "endoscopy3":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
                print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500-non-crop.pth.tar")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "endoscopy4":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
                print("loaded from /mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep500.pth.tar")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "endoscopy_mae":
                encoder = vit_base(img_size=[224])
                print(self.type_pretrained)
                pth = "/root/quanhd/ijepa_endoscopy_pretrained/jepa_continue_mae-ep400.pth.tar"
                ckpt = torch.load(pth)
                print(f"loaded from {pth} at epoch {ckpt['epoch']}")
                encoder.load_state_dict(ckpt[self.type_encoder])
            # elif self.type_pretrained == "endoscopy_mae_adapter":
            #     encoder = IJEPAAdapter(pretrain_size=224)
            #     print("Use adapter")
            #     print(self.type_pretrained)
            #     pth = "/mnt/quanhd/ijepa_stable/logs_final_mae/jepa-ep400.pth.tar"
            #     ckpt = torch.load(pth)
            #     print(f"loaded from {pth} at epoch {ckpt['epoch']}")
            #     encoder.load_state_dict(ckpt[self.type_encoder], strict=False)
            elif self.type_pretrained == "endoscopy_mae1":
                encoder = vit_base(img_size=[224])
                print(self.type_pretrained)
                pth = "/mnt/quanhd/ijepa_stable/logs_final_mae/jepa-latest.pth.tar"
                ckpt = torch.load(pth)
                print(f"loaded from {pth} at epoch {ckpt['epoch']}")
                encoder.load_state_dict(ckpt[self.type_encoder])
            elif self.type_pretrained == "mae":
                encoder = vit_base(img_size=[224])
                print(self.type_pretrained)
                pth = "/mnt/quanhd/ijepa_endoscopy_pretrained/mae_pretrain_vit_base.pth"
                ckpt = torch.load(pth)
                del ckpt["model"]["cls_token"]
                ckpt["model"]["pos_embed"] = ckpt["model"]["pos_embed"][:, 1:, :]
                encoder.load_state_dict(ckpt["model"])
            elif self.type_pretrained == "none":
                encoder = vit_base(img_size=[256])
                print(self.type_pretrained)
            elif self.type_pretrained == "im1k":
                encoder = vit_huge(img_size=[448])
                print(self.type_pretrained)
                ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_IN1K-vit.h.16-448px-300e.pth.tar")
                new_state_dict = load_state_dict_wo_module(ckpt)
                encoder.load_state_dict(new_state_dict)
            self.net = UNETRMultitask(img_size=self.img_size[0], backbone="ijepa", encoder=encoder)
            self.net.to(self.device)
        

    def init_optim(self):
        if self.type_pretrained == "resnet":
            backbone_name = "backbone"
        else:
            backbone_name = "encoder"
        base, head = [], []
        for name, param in self.net.named_parameters():
            if backbone_name in name:
                base.append(param)
            else:
                head.append(param)

        if self.type_opt == "Adam":
            self.optimizer = optim.Adam([{'params': base}, {'params': head}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.type_opt == "SGD":
            self.optimizer = optim.SGD([{'params': base}, {'params': head}], lr=3e-5, weight_decay=0, momentum=0.9)
        elif self.type_opt == "AdamW":
            self.optimizer = optim.AdamW([{'params': base}, {'params': head}], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)

        ipe = len(self.train_data_loader)
        self.lr_scheduler = WarmupCosineSchedule(self.optimizer, ipe*self.warmup_epochs/self.accum_iter, self.MIN_LR, self.BASE_LR, ipe*self.epoch_num/self.accum_iter, self.MAX_LR)

    def init_logger(self):
        name = f"{self.type_encoder}-{self.type_pretrained}-freeze:{self.num_freeze}-max_lr:{self.MAX_LR}-img_size:{self.img_size}-train_ratio:{self.train_ratio}-scale_lr:{self.scale_lr}"
        wandb.login(key=self.wandb_token)
        wandb.init(
            project="Multitask",
            name=name,
            config={
                "batch": self.batch_size,
                "MAX_LR": self.MAX_LR,
                "BASE_LR": self.BASE_LR,
                "MIN_LR": self.MIN_LR,
                "img_size": self.img_size,
                "epoch_num": self.epoch_num,
                "accum_iter": self.accum_iter
            },
        )

    def init_data_loader(self):
        train_dataset = Multitask(metadata_file=self.metadata_file, img_size=self.img_size, segmentation_classes=5, mode="train", root_path=self.root_path)
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        valid_dataset = Multitask(metadata_file=self.metadata_file, img_size=self.img_size, segmentation_classes=5, mode="test", root_path=self.root_path)
        self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False, drop_last=True)

    def run(self):
        for epoch in range(self.epoch_num):
            train_epoch_loss, train_head_lr, train_epoch_pos_loss, \
            train_epoch_dmg_loss, train_epoch_hp_loss, train_epoch_seg_loss = self.train_one_epoch(epoch)

            wandb.log({
                "train_epoch_loss": train_epoch_loss,
                "train_backbone_lr": train_head_lr,
                "train_epoch_pos_loss": train_epoch_pos_loss,
                "train_epoch_dmg_loss": train_epoch_dmg_loss,
                "train_epoch_hp_loss": train_epoch_hp_loss,
                "train_epoch_seg_loss": train_epoch_seg_loss,
            }, step=epoch)



            val_epoch_loss, val_epoch_pos_loss, val_epoch_dmg_loss, val_epoch_hp_loss, \
            val_epoch_seg_loss, val_epoch_pos_acc, val_epoch_dmg_acc, val_epoch_hp_acc, \
            val_epoch_micro_macro_ung_thu_thuc_quan, \
            val_epoch_micro_macro_viem_thuc_quan, \
            val_epoch_micro_macro_viem_loet_hoanh_ta_trang, \
            val_epoch_micro_macro_ung_thu_da_day, \
            val_epoch_micro_macro_viem_da_day, \
            val_epoch_micro_macro_polyp, \
            conf_img_pos, conf_img_dmg, conf_img_hp = self.valid_one_epoch()

            wandb.log(
                {
                    "conf_img_pos": conf_img_pos,
                    "conf_img_dmg": conf_img_dmg,
                    "conf_img_hp": conf_img_hp,
                    "valid_loss": val_epoch_loss,
                    "val_epoch_macro_dice_ung_thu_thuc_quan": val_epoch_micro_macro_ung_thu_thuc_quan[3],
                    "val_epoch_micro_dice_ung_thu_thuc_quan": val_epoch_micro_macro_ung_thu_thuc_quan[1],
                    "val_epoch_macro_dice_viem_thuc_quan": val_epoch_micro_macro_viem_thuc_quan[3],
                    "val_epoch_micro_dice_viem_thuc_quan": val_epoch_micro_macro_viem_thuc_quan[1],
                    "val_epoch_macro_dice_viem_loet_hoanh_ta_trang": val_epoch_micro_macro_viem_loet_hoanh_ta_trang[3],
                    "val_epoch_micro_dice_viem_loet_hoanh_ta_trang": val_epoch_micro_macro_viem_loet_hoanh_ta_trang[1],
                    "val_epoch_macro_dice_ung_thu_da_day": val_epoch_micro_macro_ung_thu_da_day[3],
                    "val_epoch_micro_dice_ung_thu_da_day": val_epoch_micro_macro_ung_thu_da_day[1],
                    "val_epoch_macro_dice_viem_da_day": val_epoch_micro_macro_viem_da_day[3],
                    "val_epoch_micro_dice_viem_da_day": val_epoch_micro_macro_viem_da_day[1],
                    "val_epoch_macro_dice_polyp": val_epoch_micro_macro_polyp[3],
                    "val_epoch_micro_dice_polyp": val_epoch_micro_macro_polyp[1],
                    "val_epoch_pos_loss": val_epoch_pos_loss,
                    "val_epoch_dmg_loss": val_epoch_dmg_loss,
                    "val_epoch_hp_loss": val_epoch_hp_loss,
                    "val_epoch_seg_loss": val_epoch_seg_loss,
                    "val_epoch_pos_acc": val_epoch_pos_acc,
                    "val_epoch_dmg_acc": val_epoch_dmg_acc,
                    "val_epoch_hp_acc": val_epoch_hp_acc,
                },
                step=epoch
            )

    def train_one_epoch(self, epoch):
        steps_per_epoch = len(self.train_data_loader)
        self.net.train()
        epoch_loss = AverageMeter()

        epoch_seg_loss = AverageMeter()

        epoch_pos_loss = AverageMeter()
        epoch_dmg_loss = AverageMeter()
        epoch_hp_loss = AverageMeter()

        tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
        for batch_idx, data in enumerate(tk0):
            img, mask, position_label, damage_label, segment_weight, hp_label = data
            
            img = img.float().to(self.device)
            mask = mask.float().to(self.device)
            position_label = position_label.to(self.device)
            damage_label = damage_label.to(self.device)
            segment_weight = segment_weight.to(self.device)
            hp_label = hp_label.float().to(self.device)
            with torch.cuda.amp.autocast(enabled=self.amp):
                pos_out, dmg_out, hp_out, seg_out = self.net(img)

                loss1 = self.cls_loss(pos_out, position_label)
                loss2 = self.cls_loss(dmg_out, damage_label)
                loss3 = self.bi_cls_loss(hp_out, hp_label)   
                loss4 = self.seg_loss(seg_out, mask, segment_weight)

                loss = loss1 + loss2 + loss3 + loss4

            epoch_pos_loss.update(loss1.item())
            epoch_dmg_loss.update(loss2.item())
            epoch_hp_loss.update(loss3.item())
            epoch_seg_loss.update(loss4.item())
            epoch_loss.update(loss.item())


            if self.amp:
                self.scaler.scale(loss).backward()
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(tk0)):
                    self.lr_scheduler.step()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss = loss / self.accum_iter
                loss.backward()
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(tk0)):
                    self.lr_scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # break
            self.global_step += 1
        
        if (epoch + 1) % 2 == 0:
            ckpt_path = "/root/quanhd/DoAn/unet" + f'/snapshots_{self.type_pretrained}/{epoch+1}.pth'
            print('[Saving Checkpoint:]', ckpt_path)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler._step
            }
            torch.save(checkpoint, ckpt_path)

        return epoch_loss.avg, self.optimizer.param_groups[0]["lr"], epoch_pos_loss.avg, \
        epoch_dmg_loss.avg, epoch_hp_loss.avg, epoch_seg_loss.avg


    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()
        epoch_loss = AverageMeter()

        epoch_seg_loss = AverageMeter()

        epoch_pos_loss = AverageMeter()
        epoch_dmg_loss = AverageMeter()
        epoch_hp_loss = AverageMeter()

        epoch_micro_macro_ung_thu_thuc_quan_20230620 = MicroMacroMeter(self.device, dmg_label=1)
        epoch_micro_macro_viem_thuc_quan_20230620 = MicroMacroMeter(self.device, dmg_label=2)
        epoch_micro_macro_viem_loet_hoanh_ta_trang_20230620 = MicroMacroMeter(self.device, dmg_label=3)
        epoch_micro_macro_ung_thu_da_day_20230620 = MicroMacroMeter(self.device, dmg_label=4)
        epoch_micro_macro_viem_da_day_20230620 = MicroMacroMeter(self.device, dmg_label=5)
        epoch_micro_macro_polyp = MicroMacroMeter(self.device, dmg_label=6)

        # total_pos: total number of records that have position label
        total_pos_correct = 0
        total_pos = 0

        # total_dmg: total number of records that have damage label
        total_dmg_correct = 0
        total_dmg = 0

        # total_hp: total number of records that have hp label
        total_hp_correct = 0
        total_hp = 0

        ConfMatGen = ConfMatObj(self.device)

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

                total_pos_correct += self.get_item(pos_out, position_label)
                total_dmg_correct += self.get_item(dmg_out, damage_label)
                total_hp_correct += self.get_item_binary(hp_out, hp_label)
                ConfMatGen.add(pos_out, dmg_out, hp_out, position_label, damage_label, hp_label)

                loss1 = self.cls_loss(pos_out, position_label)
                loss2 = self.cls_loss(dmg_out, damage_label)
                loss3 = self.bi_cls_loss(hp_out, hp_label)   
                loss4 = self.seg_loss(seg_out, mask, segment_weight)

                loss = loss1 + loss2 + loss3 + loss4

                loss = loss / self.accum_iter

                epoch_pos_loss.update(loss1.item())
                epoch_dmg_loss.update(loss2.item())
                epoch_hp_loss.update(loss3.item())
                epoch_seg_loss.update(loss4.item())
                epoch_loss.update(loss.item())

                epoch_micro_macro_ung_thu_thuc_quan_20230620.update(seg_out, mask, segment_weight, damage_label)
                epoch_micro_macro_viem_thuc_quan_20230620.update(seg_out, mask, segment_weight, damage_label)
                epoch_micro_macro_viem_loet_hoanh_ta_trang_20230620.update(seg_out, mask, segment_weight, damage_label)
                epoch_micro_macro_ung_thu_da_day_20230620.update(seg_out, mask, segment_weight, damage_label)
                epoch_micro_macro_viem_da_day_20230620.update(seg_out, mask, segment_weight, damage_label)
                epoch_micro_macro_polyp.update(seg_out, mask, segment_weight, damage_label)
                # break


            epoch_pos_acc = np.nan if total_pos == 0 else total_pos_correct/total_pos
            epoch_dmg_acc = np.nan if total_dmg == 0 else total_dmg_correct/total_dmg 
            epoch_hp_acc = np.nan if total_hp == 0 else total_hp_correct/total_hp

            conf_img_pos = ConfMatGen.ret_confmat("pos")
            conf_img_dmg = ConfMatGen.ret_confmat("dmg")
            conf_img_hp = ConfMatGen.ret_confmat("hp")

            conf_img_pos = wandb.Image(conf_img_pos)
            conf_img_dmg = wandb.Image(conf_img_dmg)
            conf_img_hp = wandb.Image(conf_img_hp)


            return epoch_loss.avg, epoch_pos_loss.avg, epoch_dmg_loss.avg, epoch_hp_loss.avg, \
                    epoch_seg_loss.avg, epoch_pos_acc, epoch_dmg_acc, epoch_hp_acc,\
                    epoch_micro_macro_ung_thu_thuc_quan_20230620.ret_val(), \
                    epoch_micro_macro_viem_thuc_quan_20230620.ret_val(), \
                    epoch_micro_macro_viem_loet_hoanh_ta_trang_20230620.ret_val(), \
                    epoch_micro_macro_ung_thu_da_day_20230620.ret_val(), \
                    epoch_micro_macro_viem_da_day_20230620.ret_val(), \
                    epoch_micro_macro_polyp.ret_val(), \
                    conf_img_pos, conf_img_dmg, conf_img_hp

