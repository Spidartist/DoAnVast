from trainer.trainer import Trainer
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_pretrained', help='have pretrained on which data', default="endoscopy", type=str)
    parser.add_argument('--type_damaged', help='type of damaged', required=True, type=str)
    parser.add_argument('--json_path', help='path to json file', default="/mnt/quanhd/endoscopy/ft_ton_thuong.json", type=str)
    parser.add_argument('--root_path', help='path to root folder of data', default="/home/s/DATA/", type=str)
    parser.add_argument('--gpu', help='id of GPU', default="0", type=str)
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
                json_path=args.json_path, root_path=args.root_path, wandb_token=WANDB_TOKEN,
                )
    
    trainer.run()

if __name__ == "__main__":
    run()
