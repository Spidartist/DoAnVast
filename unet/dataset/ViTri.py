from torch.utils.data import Dataset
import json
import os
import albumentations as A
import cv2
import numpy as np
import torch

class ViTri(Dataset):
    def __init__(
            self, 
            path="/mnt/quanhd/endoscopy/vi_tri_giai_phau.json", 
            mode="train",
            img_size=256,
            root_path="home/s/DATA/"
        ):
        self.path = path
        self.img_size = img_size
        self.mode = mode
        self.root_path = root_path
        self.load_data_from_json()

    def load_data_from_json(self):
        with open(self.path) as f:
            data = json.load(f)
        self.samples = []
        if self.mode == "train":
            for e in data[self.mode]:
                self.samples.append([e["image"], e["label"]])

        else:
            for e in data[self.mode]["images"]:
                self.samples.append([e, data[self.mode]["label"]])
    
    def aug(self, image):
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(self.img_size, self.img_size),])
            resized = t1(image=image)
            image = resized['image']
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

        return t(image=image)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        full_image_path = os.path.join(self.root_path, self.samples[index][0])
        image_label = self.samples[index][1]

        img = cv2.imread(full_image_path).astype(np.float32)

        augmented = self.aug(img)
        img = augmented['image']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)
        img /= 255.
        

        return img, image_label

