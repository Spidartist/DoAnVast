from trainer.trainer_multi_v2 import Trainer
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_pretrained', help='have pretrained on which data', default="endoscopy", type=str)
    parser.add_argument('--task', help='downstream task', default="segmentation", type=str)
    parser.add_argument('--json_path', help='path to json file', default="/mnt/quanhd/endoscopy/ft_ton_thuong.json", type=str)
    parser.add_argument('--root_path', help='path to root folder of data', default="/home/s/DATA/", type=str)
    parser.add_argument('--gpu', help='id of GPU', default="0", type=str)
    parser.add_argument('--type_encoder', help='type of encoder', default="target_encoder", type=str)
    parser.add_argument('--num_freeze', help='number epoch to freeze the encoder', default=0, type=int)
    parser.add_argument('--init_lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--train_ratio', help='training ratio', default=1.0, type=float)
    parser.add_argument('--img_size', help='image size', default=256, type=int)
    parser.add_argument('--batch_size', help='batch size', default=16, type=int)
    parser.add_argument('--accum_iter', help='accum_iter', default=16, type=int)
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    # parser.add_argument('--type_vit', help='type_vit', default="plain", type=str)
    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    WANDB_TOKEN = "cca12c93cb17351580e3f9fd5136347e65a3463d"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    # device = 'cpu'

    trainer = Trainer(
                device=device, type_pretrained=args.type_pretrained,
                json_path=args.json_path, root_path=args.root_path, wandb_token=WANDB_TOKEN,
                num_freeze=args.num_freeze, init_lr=args.init_lr, \
                img_size=args.img_size, batch_size=args.batch_size, accum_iter=args.accum_iter, type_encoder=args.type_encoder,
                train_ratio=args.train_ratio, amp=args.amp)
    
    trainer.run()

if __name__ == "__main__":
    run()
