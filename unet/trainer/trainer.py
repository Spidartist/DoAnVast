import wandb
from dataset.TonThuong import TonThuong
from dataset.Polyp import Polyp
from dataset.Benchmark import Benchmark
from dataset.ViTri import ViTri
from dataset.HP import HP
from loss.loss import DiceBCELoss, WeightedPosCELoss, WeightedBCELoss
from score.score import DiceScore, IoUScore
from model.vit import vit_base, vit_huge
from model.unetr import UNETR
from utils.lr import get_warmup_cosine_lr
from utils.helper import load_state_dict_wo_module, AverageMeter

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self, device, type_pretrained, type_damaged, json_path,
            root_path, wandb_token, task="segmentation", type_seg="TonThuong", type_cls="hp"
        ):
        self.device = device
        self.type_pretrained = type_pretrained
        self.type_damaged = type_damaged
        self.json_path = json_path
        self.root_path = root_path
        self.wandb_token = wandb_token
        self.batch_size = 16  # old = 16
        self.BASE_LR = 1e-6
        self.MAX_LR = 1e-3
        self.img_size = (256, 256)
        self.epoch_num = 100
        self.save_freq = 1
        self.save_path = "/logs/"
        self.warmup_epochs = 2
        self.global_step = 0

        self.task = task
        self.type_seg = type_seg
        self.type_cls = type_cls
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
        self.cls_loss = WeightedPosCELoss().cuda()
        self.bi_cls_loss = WeightedBCELoss().cuda()

    def init_score(self):
        self.dice_score = DiceScore().to(self.device)
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.iou_score = IoUScore().to(self.device)

    def init_model(self):
        if self.type_pretrained == "endoscopy":
            encoder = vit_base(img_size=[self.img_size[0]])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/jepa-ep300.pth.tar")
            encoder.load_state_dict(ckpt["target_encoder"])
        else:
            encoder = vit_huge(img_size=[self.img_size[0]])
            print(self.type_pretrained)
            ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/IN1K-vit.h.16-448px-300e.pth.tar")
            new_state_dict = load_state_dict_wo_module(ckpt["target_encoder"])
            encoder.load_state_dict(new_state_dict)

        self.net = UNETR(img_size=self.img_size[0], backbone="ijepa", encoder=encoder)
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
            project=self.type_seg,
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
            if self.type_cls == "hp":
                train_dataset = HP(root_path=self.root_path, mode="train", img_size=self.img_size[0])
                self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

                valid_dataset = HP(root_path=self.root_path, mode="test", img_size=self.img_size[0])
                self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
            elif self.type_cls == "vitri": # undone
                train_dataset = ViTri(root_path=self.root_path, mode="train", img_size=self.img_size[0])
                self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

                valid_dataset = ViTri(root_path=self.root_path, mode="test", img_size=self.img_size[0])
                self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self):
        for epoch in range(self.epoch_num):
            if self.task == "segmentation":
                train_epoch_loss, train_epoch_dice_score, train_epoch_iou_score = self.train_one_epoch()
                wandb.log(
                        {
                            "train_epoch_loss": train_epoch_loss,
                            "train_epoch_dice_score": train_epoch_dice_score,
                            "train_epoch_iou_score": train_epoch_iou_score
                        },
                        step=epoch 
                    )
                if self.type_seg != "benchmark":
                    valid_epoch_loss, valid_epoch_dice_score, valid_epoch_iou_score, vis_image = self.valid_one_epoch()

                    wandb.log(
                        {
                            "valid_epoch_loss": valid_epoch_loss,
                            "valid_epoch_dice_score": valid_epoch_dice_score,
                        "valid_epoch_iou_score": valid_epoch_iou_score,
                            "valid_image_visualize": vis_image
                        },
                        step=epoch
                    )
                else:
                    for type_test_ds in self.valid_data_loaders:
                        self.valid_data_loader = self.valid_data_loaders[type_test_ds]
                        valid_epoch_loss, valid_epoch_score, vis_image = self.valid_one_epoch()

                        wandb.log(
                            {
                                f"{type_test_ds}_valid_epoch_loss": valid_epoch_loss,
                                f"{type_test_ds}_valid_epoch_dice_score": valid_epoch_score,
                                f"{type_test_ds}_valid_image_visualize": vis_image
                            },
                            step=epoch
                        )
            elif self.task == "classification":
                pass
            

    def train_one_epoch(self):
        steps_per_epoch = len(self.train_data_loader)
        total_steps = steps_per_epoch * self.epoch_num
        self.net.train()
        if self.task == "segmentation":
            epoch_loss = AverageMeter()
            epoch_dice_score = AverageMeter()
            epoch_iou_score = AverageMeter()

            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for data in tk0:
                img, mask = data
                n = img.shape[0]

                img = img.float().to(self.device)
                mask = mask.float().to(self.device)

                lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
                self.optimizer.param_groups[0]['lr'] = 0.1 * lr
                self.optimizer.param_groups[1]['lr'] = lr

                seg_out = self.net(img)

                loss3 = self.seg_loss(seg_out, mask, 1)

                epoch_loss.update(loss3.item(), n=n)

                self.optimizer.zero_grad()
                loss3.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.net.eval()
                    # score = dice_score(seg_out, mask[1], segment_weight)
                    dice_score = self.dice_score(seg_out, mask, 1)
                    iou_score = self.iou_score(seg_out, mask, 1)
                    epoch_dice_score.update(dice_score.item(), n=n)
                    epoch_iou_score.update(iou_score.item(), n=n)

                # if global_step % self.save_freq == 0 or global_step == total_steps-1:
                #     torch.save(self.net.state_dict(), self.save_path + f'/model-{self.type_pretrained}-{self.type_damaged}-best.pt')

                self.global_step += 1

            return epoch_loss.avg, epoch_dice_score.avg, epoch_iou_score.avg
        elif self.task == "classification":
            epoch_loss = AverageMeter()
            epoch_fi_score = AverageMeter()
            epoch_acc_score = AverageMeter()

            tk0 = tqdm(self.train_data_loader, total=steps_per_epoch)
            for data in tk0:
                img, label = data
                n = img.shape[0]

                img = img.float().to(self.device)
                label = label.float().to(self.device)

                lr = get_warmup_cosine_lr(self.BASE_LR, self.MAX_LR, self.global_step, total_steps, steps_per_epoch, warmup_epochs=self.warmup_epochs)
                self.optimizer.param_groups[0]['lr'] = 0.1 * lr
                self.optimizer.param_groups[1]['lr'] = lr

                cls_out = self.net(img)

                if self.type_cls == "vitri":
                    loss3 = self.cls_loss(cls_out, label, 1)
                elif self.type_cls == "HP":
                    loss3 = self.bi_cls_loss(cls_out, label, 1)

                epoch_loss.update(loss3.item(), n=n)

                self.optimizer.zero_grad()
                loss3.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.net.eval()
                    f1_score = self.f1_score(cls_out, label, 1)
                    acc_score = self.accuracy(cls_out, label, 1)
                    epoch_dice_score.update(dice_score.item(), n=n)
                    epoch_iou_score.update(iou_score.item(), n=n)


                self.global_step += 1


    def valid_one_epoch(self):
        steps_per_epoch = len(self.valid_data_loader)
        self.net.eval()
        epoch_loss = AverageMeter()
        epoch_dice_score = AverageMeter()
        epoch_iou_score = AverageMeter()

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

                dice_score = self.dice_score(seg_out, mask, 1)
                epoch_dice_score.update(dice_score.item(), n=n)

                iou_score = self.iou_score(seg_out, mask, 1)
                epoch_iou_score.update(iou_score.item(), n=n)

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
        return epoch_loss.avg , epoch_dice_score.avg, epoch_iou_score.avg, vis_image

    
