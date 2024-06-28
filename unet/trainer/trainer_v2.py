import wandb
from dataset.Multitask import Multitask
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss, structure_loss
from score.score import DiceScore, IoUScore, MicroMacroDiceIoU
from model.vit import vit_base, vit_huge
from model.RaBiT import RaBiTSegmentor
from dataset.Benchmark import Benchmark
from dataset.TonThuong import TonThuong
from dataset.Polyp import Polyp
from model.utils import Feature2Pyramid
from utils.helper import load_state_dict_wo_module, AverageMeter, ScoreAverageMeter, GetItem, GetItemBinary
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
        self.dice_IoU = MicroMacroDiceIoU().to(self.device)
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
            project="benchmark_rabit",
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
            train_epoch_loss, train_head_lr = self.train_one_epoch(epoch)

            wandb.log({
                "train_epoch_loss": train_epoch_loss,
                "train_backbone_lr": train_head_lr,
            }, step=epoch)

            if self.type_seg != "benchmark":
                valid_epoch_loss, micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image = self.valid_one_epoch()
                wandb.log(
                    {
                        "valid_loss": valid_epoch_loss,
                        "valid_micro_dice_score": micro_dice_score,
                        "valid_micro_iou_score": micro_iou_score,
                        "valid_macro_dice_score": macro_dice_score,
                        "valid_macro_iou_score": macro_iou_score,
                        "valid_image_visualize": vis_image
                    },
                    step=epoch
                )
            else:
                for type_test_ds in self.valid_data_loaders:
                    self.valid_data_loader = self.valid_data_loaders[type_test_ds]
                    valid_epoch_loss, micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image = self.valid_one_epoch()

                    wandb.log(
                        {
                            f"{type_test_ds}_valid_loss": valid_epoch_loss,
                            f"{type_test_ds}_valid_micro_dice_score": micro_dice_score,
                            f"{type_test_ds}_valid_micro_iou_score": micro_iou_score,
                            f"{type_test_ds}_valid_macro_dice_score": macro_dice_score,
                            f"{type_test_ds}_valid_macro_iou_score": macro_iou_score,
                            f"{type_test_ds}_valid_image_visualize": vis_image
                        },
                        step=epoch
                    )

    def train_one_epoch(self, epoch):
        steps_per_epoch = len(self.train_data_loader)
        self.net.train()
        epoch_loss = AverageMeter()

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for i, data in enumerate(tqdm(self.train_data_loader), start=1):
                if epoch <= self.warmup_epochs:
                    self.optimizer.param_groups[0]["lr"] = self.init_lr * (i / steps_per_epoch + epoch - 1) / self.warmup_epochs
                else:
                    self.lr_scheduler.step()

                img, mask = data
                
                img = img.float().to(self.device)
                mask = mask.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.net(img)
                    seg_loss1 = structure_loss(output["map"][0], mask)
                    seg_loss2 = structure_loss(output["map"][1], mask)
                    seg_loss3 = structure_loss(output["map"][2], mask)
                    seg_loss4 = structure_loss(output["map"][3], mask)
                    seg_loss = seg_loss1 + seg_loss2 + seg_loss3 + seg_loss4

                    loss = seg_loss

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

        return epoch_loss.avg, self.optimizer.param_groups[0]["lr"]


    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()
        epoch_loss = AverageMeter()

        epoch_dice_score = ScoreAverageMeter(self.device)
        epoch_iou_score = ScoreAverageMeter(self.device)
        epoch_intersection = ScoreAverageMeter(self.device)
        epoch_union = ScoreAverageMeter(self.device)
        epoch_intersection2 = ScoreAverageMeter(self.device)
        epoch_total_area = ScoreAverageMeter(self.device)

        vis_image = None

        tk0 = tqdm(self.valid_data_loader, total=steps_per_epoch)
        with torch.no_grad():
            for data in tk0:
                img, mask = data

                img = img.float().to(self.device)
                mask = mask.float().to(self.device)

                output = self.net(img)

                seg_loss1 = structure_loss(output["map"][0], mask)
                seg_loss2 = structure_loss(output["map"][1], mask)
                seg_loss3 = structure_loss(output["map"][2], mask)
                seg_loss4 = structure_loss(output["map"][3], mask)
                seg_loss = seg_loss1 + seg_loss2 + seg_loss3 + seg_loss4

                loss = seg_loss

                loss = loss / self.accum_iter

                epoch_loss.update(loss.item())
                seg_out = output["map"][0]
                iou, dice, intersection, union, intersection2, total_area = self.dice_IoU(seg_out, mask)
                epoch_iou_score.update(iou.to(self.device))
                epoch_dice_score.update(dice.to(self.device))
                epoch_intersection.update(intersection.to(self.device))
                epoch_union.update(union.to(self.device))
                epoch_intersection2.update(intersection2.to(self.device))
                epoch_total_area.update(total_area.to(self.device))

                seg_img = seg_out[0]
                seg_img[seg_img <= 0.5] = 0
                seg_img[seg_img > 0.5] = 1

                images = torch.cat([seg_img, mask[0]], dim=1)
                images = wandb.Image(images)
                vis_image = images
                # break

            micro_iou_score = epoch_intersection.lst_tensor.sum()/epoch_union.lst_tensor.sum()
            micro_dice_score = epoch_intersection2.lst_tensor.sum()/epoch_total_area.lst_tensor.sum()
            macro_iou_score = epoch_iou_score.lst_tensor.mean()
            macro_dice_score = epoch_dice_score.lst_tensor.mean()
            # if epoch_loss < best_epoch_loss:
            #     print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
            #     best_epoch_loss = epoch_loss
            #     torch.save(self.net.state_dict(), self.save_path + f'/model-{type}-{loai_ton_thuong}-best.pt')
            return epoch_loss.avg, micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image

