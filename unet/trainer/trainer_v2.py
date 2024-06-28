import wandb
from dataset.Multitask import Multitask
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss, structure_loss
from score.score import DiceScore, IoUScore, MicroMacroDiceIoUMultitask
from model.vit import vit_base, vit_huge
from model.RaBiT import RaBiTSegmentor
from dataset.Benchmark import Benchmark
from dataset.TonThuong import TonThuong
from dataset.Polyp import Polyp
from model.utils import Feature2Pyramid
from utils.helper import load_state_dict_wo_module, AverageMeter, MicroMacroMeter, GetItem, GetItemBinary, ConfMatObj
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, json_path, amp,
            wandb_token,  type_damaged, init_lr=1e-4,
            num_freeze=10, img_size=256, batch_size=16, accum_iter=16,
            type_encoder="target_encoder", train_ratio=1.0, metadata_file="/mnt/quanhd/test/DoAnVast/unet/dataset/data_dir_endounet.json",
            continue_ckpt="", root_path="/mnt/tuyenld/data/endoscopy/", type_seg="TonThuong",
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.metadata_file = metadata_file
        self.type_damaged = type_damaged
        self.type_seg = type_seg
        self.json_path = json_path
        self.root_path = root_path
        self.continue_ckpt = continue_ckpt
        self.num_freeze= num_freeze
        self.wandb_token = wandb_token
        self.accum_iter = accum_iter
        self.type_encoder = type_encoder
        self.init_lr = init_lr
        self.amp = amp
        self.train_ratio = train_ratio
        self.img_size = (img_size, img_size)
        self.batch_size = batch_size
        self.epoch_num = 20
        self.save_freq = 1
        self.save_path = "/logs/"
        self.warmup_epochs = 2
        self.global_step = 0

        self.init_data_loader()
        self.init_loss()
        self.init_score()
        self.init_model()
        self.init_optim()
        self.init_logger()
        self.display_info()

    def display_info(self):
        print('epoch            : %d' % self.epoch_num)
        print('batch_size       : %d' % self.batch_size)
        print('save_freq        : %d' % self.save_freq)
        print('img_size         : (%d, %d)' % (self.img_size[0], self.img_size[1]))
        print('init_lr          : %s' % self.init_lr)
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
            pth = "/mnt/quanhd/ijepa_stable/logs_final_mae/jepa-ep400.pth.tar"
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
        neck = Feature2Pyramid()
        self.net = RaBiTSegmentor(backbone=encoder, neck=neck)
        self.net.to(self.device)
        

    def init_optim(self):
        params = self.net.parameters()
        self.optimizer = torch.optim.Adam(params, self.init_lr)
        
        ipe = len(self.train_data_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                    T_max=ipe*self.epoch_num,
                                    eta_min=self.init_lr/1000)
        
    def init_logger(self):
        name = f"{self.type_encoder}-{self.type_pretrained}-freeze:{self.num_freeze}-init_lr:{self.init_lr}-img_size:{self.img_size}-train_ratio:{self.train_ratio}"
        wandb.login(key=self.wandb_token)
        wandb.init(
            project="Multitask1",
            name=name,
            config={
                "batch": self.batch_size,
                "init_lr": self.init_lr,
                "img_size": self.img_size,
                "epoch_num": self.epoch_num,
                "accum_iter": self.accum_iter
            },
        )

    def init_data_loader(self):
        if self.type_seg == "TonThuong":
            train_dataset = TonThuong(root_path=self.root_path, mode="train", type=self.type_damaged, img_size=self.img_size[0])
            self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

            valid_dataset = TonThuong(root_path=self.root_path, mode="test", type=self.type_damaged, img_size=self.img_size[0])
            self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
        elif self.type_seg == "polyp":
            train_dataset = Polyp(root_path=self.root_path, mode="train", img_size=self.img_size[0])
            self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

            valid_dataset = Polyp(root_path=self.root_path, mode="test", img_size=self.img_size[0])
            self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
        elif self.type_seg == "benchmark":
            path = "/mnt/quanhd/endoscopy/public_dataset.json"
            train_dataset = Benchmark(path=path, root_path=self.root_path, img_size=self.img_size[0], train_ratio=self.train_ratio)
            self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            
            self.valid_data_loaders = {}
            valid_dataset = Benchmark(path=path, root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-300")
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

            valid_dataset = Benchmark(path=path, root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="Kvasir")
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

            valid_dataset = Benchmark(path=path, root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-ClinicDB")
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

            valid_dataset = Benchmark(path=path, root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-ColonDB")
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

            valid_dataset = Benchmark(path=path, root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="ETIS-LaribPolypDB")
            valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader
    def run(self):
        for epoch in range(self.epoch_num):
            train_epoch_loss, train_head_lr, train_epoch_seg_loss = self.train_one_epoch(epoch)

            wandb.log({
                "train_epoch_loss": train_epoch_loss,
                "train_backbone_lr": train_head_lr,
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

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for i, data in enumerate(tqdm(self.train_data_loader), start=1):
                if epoch <= self.warmup_epochs:
                    self.optimizer.param_groups[0]["lr"] = self.init_lr * (i / steps_per_epoch + epoch - 1) / self.warmup_epochs
                else:
                    self.lr_scheduler.step()

                img, mask, position_label, damage_label, segment_weight, hp_label = data
                
                img = img.float().to(self.device)
                mask = mask.float().to(self.device)
                position_label = position_label.to(self.device)
                damage_label = damage_label.to(self.device)
                segment_weight = segment_weight.to(self.device)
                hp_label = hp_label.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.net(img)
                    seg_loss1 = structure_loss(output["map"][0], mask, segment_weight)
                    seg_loss2 = structure_loss(output["map"][1], mask, segment_weight)
                    seg_loss3 = structure_loss(output["map"][2], mask, segment_weight)
                    seg_loss4 = structure_loss(output["map"][3], mask, segment_weight)
                    seg_loss = seg_loss1 + seg_loss2 + seg_loss3 + seg_loss4

                    loss = seg_loss

                epoch_seg_loss.update(seg_loss.item())
                epoch_loss.update(loss.item())


                if self.amp:
                    self.scaler.scale(loss).backward()
                    if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(tk0)):
                        # self.lr_scheduler.step()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss = loss / self.accum_iter
                    loss.backward()
                    if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(tk0)):
                        self.lr_scheduler.step()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                # break
                self.global_step += 1
            
            # if (epoch + 1) % 2 == 0:
            #     ckpt_path = "/root/quanhd/DoAn/unet" + f'/snapshots_rabit/{epoch+1}.pth'
            #     print('[Saving Checkpoint:]', ckpt_path)
            #     checkpoint = {
            #         'epoch': epoch + 1,
            #         'state_dict': self.net.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #         'scheduler': self.lr_scheduler.state_dict()
            #     }
            #     torch.save(checkpoint, ckpt_path)

        return epoch_loss.avg, self.optimizer.param_groups[0]["lr"], epoch_seg_loss.avg


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

                output = self.net(img)

                total_pos_correct += self.get_item(output["pos"], position_label)
                total_dmg_correct += self.get_item(output["type"], damage_label)
                total_hp_correct += self.get_item_binary(output["hp"], hp_label)
                ConfMatGen.add(output["pos"], output["type"], output["hp"], position_label, damage_label, hp_label)

                seg_loss1 = structure_loss(output["map"][0], mask, segment_weight)
                seg_loss2 = structure_loss(output["map"][1], mask, segment_weight)
                seg_loss3 = structure_loss(output["map"][2], mask, segment_weight)
                seg_loss4 = structure_loss(output["map"][3], mask, segment_weight)
                seg_loss = seg_loss1 + seg_loss2 + seg_loss3 + seg_loss4

                loss = seg_loss

                loss = loss / self.accum_iter

                epoch_seg_loss.update(seg_loss.item())
                epoch_loss.update(loss.item())

                # print(output["map"][0].shape)

                epoch_micro_macro_ung_thu_thuc_quan_20230620.update(output["map"][0], mask, segment_weight, damage_label)
                epoch_micro_macro_viem_thuc_quan_20230620.update(output["map"][0], mask, segment_weight, damage_label)
                epoch_micro_macro_viem_loet_hoanh_ta_trang_20230620.update(output["map"][0], mask, segment_weight, damage_label)
                epoch_micro_macro_ung_thu_da_day_20230620.update(output["map"][0], mask, segment_weight, damage_label)
                epoch_micro_macro_viem_da_day_20230620.update(output["map"][0], mask, segment_weight, damage_label)
                epoch_micro_macro_polyp.update(output["map"][0], mask, segment_weight, damage_label)
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

