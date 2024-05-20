from trainer.trainer import Trainer
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_pretrained', help='have pretrained on which data', default="endoscopy", type=str)
    parser.add_argument('--type_damaged', help='type of damaged', default="ung_thu_da_day_20230620", type=str)
    parser.add_argument('--type_seg', help='type of segmentation', default="TonThuong", type=str)
    parser.add_argument('--type_cls', help='type of classification', default="HP", type=str)
    parser.add_argument('--task', help='downstream task', default="segmentation", type=str)
    parser.add_argument('--json_path', help='path to json file', default="/mnt/quanhd/endoscopy/ft_ton_thuong.json", type=str)
    parser.add_argument('--root_path', help='path to root folder of data', default="/home/s/DATA/", type=str)
    parser.add_argument('--gpu', help='id of GPU', default="0", type=str)
    parser.add_argument('--num_freeze', help='number epoch to freeze the encoder', default=0, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    WANDB_TOKEN = "cca12c93cb17351580e3f9fd5136347e65a3463d"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    trainer = Trainer(
                device=device, type_pretrained=args.type_pretrained, type_damaged=args.type_damaged,
                json_path=args.json_path, root_path=args.root_path, wandb_token=WANDB_TOKEN, type_seg=args.type_seg,
                num_freeze=args.num_freeze, max_lr=args.lr, task=args.task, type_cls=args.type_cls)
    
    trainer.run()

if __name__ == "__main__":
    run()
