from torch.utils.data import Dataset, DataLoader
import json
from pprint import pprint
import os
import albumentations as A
from albumentations import augmentations
import cv2
import numpy as np
import torch

class Polyp(Dataset):
    def __init__(
            self, 
            path="/mnt/quanhd/endoscopy/polyp.json", 
            mode="train",
            img_size=256,
            root_path="home/s/DATA/"
        ):
        self.path = path
        self.img_size = img_size
        self.mode = mode
        self.root_path = root_path
        self.load_data_from_json()
        # pprint(self.image_paths)

    def load_data_from_json(self):
        with open(self.path) as f:
            data = json.load(f)
        self.image_paths = data[self.mode]["images"]
        # print(len(self.image_paths))
        self.mask_paths = data[self.mode]["masks"]
    
    def aug(self, image, mask):
        img_size = self.img_size
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(img_size, img_size),])
            resized = t1(image=image, mask=mask)
            image = resized['image']
            mask = resized['mask']
            t = A.Compose([                
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.Rotate(interpolation=cv2.BORDER_CONSTANT, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=0, p=0.35),
                A.MotionBlur(p=0.2),
                A.HueSaturationValue(p=0.2),                
            ], p=0.5)

        elif self.mode == 'test':
            t = A.Compose([
                A.Resize(img_size, img_size)
            ])

        return t(image=image, mask=mask)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        full_image_path = os.path.join(self.root_path, self.image_paths[index])
        full_mask_path = os.path.join(self.root_path, self.mask_paths[index])

        img = cv2.imread(full_image_path).astype(np.float32)
        orin_mask = cv2.imread(full_mask_path).astype(np.float32)

        augmented = self.aug(img, orin_mask)
        img = augmented['image']
        orin_mask = augmented['mask']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)
        img /= 255.
        
        orin_mask = torch.from_numpy(orin_mask.copy())
        orin_mask = orin_mask.permute(2, 0, 1)
        orin_mask = orin_mask.mean(dim=0, keepdim=True)/255.
        orin_mask[orin_mask <= 0.5] = 0
        orin_mask[orin_mask > 0.5] = 1

        return img, orin_mask



        



if __name__ == "__main__":
    ds = TonThuong()
    print(ds.__len__())