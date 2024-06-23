from torch.utils.data import Dataset, DataLoader
import json
from pprint import pprint
import os
import albumentations as A
from albumentations import augmentations
import cv2
import numpy as np
import torch

class Benchmark(Dataset):
    def __init__(
            self, 
            path="/mnt/quanhd/endoscopy/public_dataset.json", 
            mode="train",
            ds_test="CVC-300",
            img_size=256,
            root_path="home/s/DATA/",
            train_ratio=1.0
            # root_path="/mnt/tuyenld/data/endoscopy/"

        ):
        self.path = path
        self.img_size = img_size
        self.mode = mode
        self.ds_test = ds_test
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.load_data_from_json()

    def load_data_from_json(self):
        with open(self.path) as f:
            data = json.load(f)
        if self.mode == "train":
            all_image_paths = data[self.mode]["images"]
            kvasir_image_paths = []
            clinic_image_paths = []
            for image_path in all_image_paths:
                if "c" in image_path:
                    kvasir_image_paths.append(image_path)
                else:
                    clinic_image_paths.append(image_path)
        
            all_mask_paths = data[self.mode]["masks"]
            kvasir_mask_paths = []
            clinic_mask_paths = []
            for mask_path in all_mask_paths:
                if "c" in mask_path:
                    kvasir_mask_paths.append(mask_path)
                else:
                    clinic_mask_paths.append(mask_path)
            print(f"Pre len(all_image_paths) =s {len(all_image_paths)}")
            print(f"Pre len(all_mask_paths) = {len(all_mask_paths)}")
            self.image_paths = kvasir_image_paths[:int(len(kvasir_image_paths)*self.train_ratio)] + clinic_image_paths[:int(len(clinic_image_paths)*self.train_ratio)]
            self.mask_paths = kvasir_mask_paths[:int(len(kvasir_mask_paths)*self.train_ratio)] + clinic_mask_paths[:int(len(clinic_mask_paths)*self.train_ratio)]
            print(f"After len(image_paths) = {len(self.image_paths)}")
            print(f"After len(mask_paths) = {len(self.mask_paths)}")
        elif self.mode == "test":
            self.image_paths = data[self.mode][self.ds_test]["images"]
            self.mask_paths = data[self.mode][self.ds_test]["masks"]
    
    def aug(self, image, mask):
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(self.img_size, self.img_size),])
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
                A.Resize(self.img_size, self.img_size)
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
    ds = Benchmark()
    print(ds.image_paths[5])
    print(ds.mask_paths[5])

