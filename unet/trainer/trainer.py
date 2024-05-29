import wandb
from dataset.TonThuong import TonThuong
from dataset.Polyp import Polyp
from dataset.Benchmark import Benchmark
from dataset.ViTri import ViTri
from dataset.HP import HP
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss
from score.score import DiceScore, IoUScore, MicroMacroDiceIoU
from model.vit import vit_base, vit_huge
from model.unetr import UNETR
from utils.lr import get_warmup_cosine_lr
from utils.helper import load_state_dict_wo_module, AverageMeter, ScoreAverageMeter, GetItem, GetItemBinary

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, type_damaged, json_path,
            root_path, wandb_token, task="segmentation", type_seg="TonThuong", type_cls="HP",
            num_freeze=10, max_lr=1e-3, img_size=256, type_opt="Adam", batch_size=16, accum_iter=16
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.type_damaged = type_damaged
        self.json_path = json_path
        self.root_path = root_path
        self.num_freeze= num_freeze
        self.wandb_token = wandb_token
        self.accum_iter = accum_iter
        self.BASE_LR = 1e-6
        self.type_opt = type_opt
        self.MAX_LR = max_lr
        self.img_size = (img_size, img_size)
        self.batch_size = batch_size
        # if self.type_pretrained == "endoscopy" or self.type_pretrained == "endoscopy1" or self.type_pretrained == "none" or self.type_pretrained == "endoscopy2" or self.type_pretrained == "endoscopy3":
        #     # self.img_size = (256, 256)
        #     self.batch_size = 16  # old = 16
        # elif self.type_pretrained == "im1k":
        #     # self.img_size = (448, 448)
        #     self.batch_size = 1
        self.epoch_num = 100
        self.save_freq = 1
        self.save_path = "/logs/"
        self.warmup_epochs = 2
        self.global_step = 0

        self.task = task
        self.type_seg = type_seg
        self.type_cls = type_cls
        # self.init_logger()
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
        print('BASE_LR          : %s' % self.BASE_LR)
        print('MAX_LR           : %s' % self.MAX_LR)
        print('warmup_epochs:   : %d' % self.warmup_epochs)
        print('device           : %s' % self.device)
        print('log dir          : %s' % self.save_path)
        print('model has {} parameters in total'.format(sum(x.numel() for x in self.net.parameters())))

    def init_loss(self):
        self.seg_loss = DiceBCELoss().to(self.device)
        self.cls_loss = WeightedPosCELoss().cuda()
        self.bi_cls_loss = WeightedBCELoss().cuda()

    def init_score(self):
        self.dice_score = DiceScore().to(self.device)
        self.precision = None
        self.recall = None
        self.f1_score = None
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
        if self.task == "segmentation":
            self.net = UNETR(img_size=self.img_size[0], backbone="ijepa", encoder=encoder)
        elif self.task == "classification":
            if self.type_cls == "HP":
                self.net = UNETR(img_size=self.img_size[0], backbone="ijepa", encoder=encoder, task="classification", type_cls="HP")
            elif self.type_cls == "vitri":
                self.net = UNETR(img_size=self.img_size[0], backbone="ijepa", encoder=encoder, task="classification", type_cls="vitri")
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

        if self.type_opt == "Adam":
            self.optimizer = optim.Adam([{'params': base}, {'params': head}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.type_opt == "SGD":
            self.optimizer = optim.SGD([{'params': base}, {'params': head}], lr=0.001, weight_decay=0)
        elif self.type_opt == "AdamW":
            self.optimizer = optim.AdamW([{'params': base}, {'params': head}], lr=1e-4, weight_decay=1e-5)

    def init_logger(self):
        if self.task == "segmentation":
            name = f"{self.type_opt}-{self.type_seg}-{self.type_damaged}-{self.type_pretrained}-freeze:{self.num_freeze}-max_lr:{self.MAX_LR}-img_size:{self.img_size}"
        elif self.task == "classification":
            name = f"{self.type_opt}-{self.type_cls}-{self.type_pretrained}-freeze:{self.num_freeze}-max_lr:{self.MAX_LR}-img_size:{self.img_size}"
        wandb.login(key=self.wandb_token)
        wandb.init(
            project=self.type_seg+"2",
            name=name,
            config={
                "batch": self.batch_size,
                "MAX_LR": self.MAX_LR,
                "BASE_LR": self.BASE_LR,
                "img_size": self.img_size,
                "epoch_num": self.epoch_num,
                "accum_iter": self.accum_iter
            },
        )

    def init_data_loader(self):
        if self.task == "segmentation":
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
                train_dataset = Benchmark(root_path=self.root_path, img_size=self.img_size[0])
                self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
                
                self.valid_data_loaders = {}
                valid_dataset = Benchmark(root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-300")
                valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

                valid_dataset = Benchmark(root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="Kvasir")
                valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

                valid_dataset = Benchmark(root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-ClinicDB")
                valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

                valid_dataset = Benchmark(root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="CVC-ColonDB")
                valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

                valid_dataset = Benchmark(root_path=self.root_path, mode="test", img_size=self.img_size[0], ds_test="ETIS-LaribPolypDB")
                valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.valid_data_loaders[valid_dataset.ds_test] = valid_data_loader

        elif self.task == "classification":
            if self.type_cls == "HP":
                train_dataset = HP(root_path=self.root_path, mode="train", img_size=self.img_size[0])
                self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

                valid_dataset = HP(root_path=self.root_path, mode="test", img_size=self.img_size[0])
                self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            elif self.type_cls == "vitri": 
                train_dataset = ViTri(root_path=self.root_path, mode="train", img_size=self.img_size[0])
                self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

                valid_dataset = ViTri(root_path=self.root_path, mode="test", img_size=self.img_size[0])
                self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self):
        for epoch in range(self.epoch_num):
            if epoch == self.num_freeze:
                self.net.unfreeze_encoder()
            if self.task == "segmentation":
                train_epoch_loss = self.train_one_epoch()
                wandb.log(
                        {
                            "train_epoch_loss": train_epoch_loss,
                        },
                        step=epoch 
                    )
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
            elif self.task == "classification":
                train_epoch_loss = self.train_one_epoch()
                wandb.log(
                    {
                        "train_epoch_loss": train_epoch_loss,
                    },
                    step=epoch 
                )
                valid_epoch_loss, valid_accuracy = self.valid_one_epoch()

                wandb.log(
                    {
                        "valid_loss": valid_epoch_loss,
                        "valid_accuracy": valid_accuracy,
                    },
                    step=epoch
                )
            

    def train_one_epoch(self):
        steps_per_epoch = len(self.train_data_loader)
        total_steps = steps_per_epoch * self.epoch_num
        self.net.train()
        if self.task == "segmentation":
            epoch_loss = AverageMeter()

            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for batch_idx, data in enumerate(tk0):
                img, mask = data
                n = img.shape[0]

                img = img.float().to(self.device)
                mask = mask.float().to(self.device)

                lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
                self.optimizer.param_groups[0]['lr'] = 0.1 * lr
                self.optimizer.param_groups[1]['lr'] = lr

                seg_out = self.net(img)

                loss3 = self.seg_loss(seg_out, mask, 1)

                loss3 = loss3 / self.accum_iter


                epoch_loss.update(loss3.item(), n=n)
                loss3.backward()

                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(tk0)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # if global_step % self.save_freq == 0 or global_step == total_steps-1:
                #     torch.save(self.net.state_dict(), self.save_path + f'/model-{self.type_pretrained}-{self.type_damaged}-best.pt')

                self.global_step += 1

            return epoch_loss.avg
        elif self.task == "classification":
            epoch_loss = AverageMeter()
            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for data in tk0:
                img, label = data
                n = img.shape[0]

                img = img.float().to(self.device)
                label = label.to(self.device)

                lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
                self.optimizer.param_groups[0]['lr'] = 0.1 * lr
                self.optimizer.param_groups[1]['lr'] = lr

                cls_out = self.net(img)

                if self.type_cls == "vitri":
                    loss3 = self.cls_loss(cls_out, label)
                elif self.type_cls == "HP":
                    label = label.float()
                    loss3 = self.bi_cls_loss(cls_out, label)

                epoch_loss.update(loss3.item(), n=n)

                self.optimizer.zero_grad()
                loss3.backward()
                self.optimizer.step()

                self.global_step += 1
            return epoch_loss.avg


    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()
        if self.task == "segmentation":
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
                    n = img.shape[0]

                    img = img.float().to(self.device)
                    mask = mask.float().to(self.device)

                    seg_out = self.net(img)

                    loss3 = self.seg_loss(seg_out, mask, 1)

                    epoch_loss.update(loss3.item(), n=n)

                    iou, dice, intersection, union, intersection2, total_area = self.dice_IoU(seg_out, mask)
                    # print(iou)
                    # print(intersection)
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

            micro_iou_score = epoch_intersection.lst_tensor.sum()/epoch_union.lst_tensor.sum()
            micro_dice_score = epoch_intersection2.lst_tensor.sum()/epoch_total_area.lst_tensor.sum()
            macro_iou_score = epoch_iou_score.lst_tensor.mean()
            macro_dice_score = epoch_dice_score.lst_tensor.mean()
            # if epoch_loss < best_epoch_loss:
            #     print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
            #     best_epoch_loss = epoch_loss
            #     torch.save(self.net.state_dict(), self.save_path + f'/model-{type}-{loai_ton_thuong}-best.pt')
            return epoch_loss.avg, micro_dice_score, micro_iou_score, macro_dice_score, macro_iou_score, vis_image
        elif self.task == "classification":
            epoch_loss = AverageMeter()
            get_item = GetItem()
            get_item_binary = GetItemBinary()
            # epoch_f1_score = AverageMeter()
            # epoch_acc_score = AverageMeter()
            total_sample = 0
            total_correct_sample = 0

            tk0 = tqdm(self.valid_data_loader, total=steps_per_epoch)
            with torch.no_grad():
                for data in tk0:
                    img, label = data
                    n = img.shape[0]
                    total_sample += n
                    img = img.float().to(self.device)
                    label = label.to(self.device)

                    cls_out = self.net(img)

                    if self.type_cls == "vitri":
                        loss3 = self.cls_loss(cls_out, label)
                        total_correct_sample += get_item(cls_out, label)
                    elif self.type_cls == "HP":
                        label = label.float()
                        loss3 = self.bi_cls_loss(cls_out, label)
                        total_correct_sample += get_item_binary(cls_out, label)

                    epoch_loss.update(loss3.item(), n=n)
            acc = total_correct_sample/total_sample
            return epoch_loss.avg, acc

