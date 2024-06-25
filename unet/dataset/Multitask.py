from torch.utils.data import Dataset, DataLoader
import json
from pprint import pprint
import os
import albumentations as A
from albumentations import augmentations
import cv2
import numpy as np
import torch
import random

class Multitask(Dataset):
    """
    Data loader for binary-segmentation training
    """
    def __init__(self, root_path="/root/quanhd/DATA/", metadata_file='/root/quanhd/DoAn/unet/dataset/data_dir_endounet.json', img_size=(320, 320), segmentation_classes=5, mode="train"):
        self.train_samples = []
        self.test_samples = []
        self.img_size = img_size
        self.root_path = root_path
        self.segmentation_classes = segmentation_classes
        self.mode = mode

        with open("/root/quanhd/endoscopy/hp.json") as f:
            hp_data = json.load(f)

        with open(metadata_file) as f:
            dirs = json.load(f)['dirs']

        for dir_info in dirs:
            type = dir_info['type']
            position_label = dir_info.get('position_label', -1)
            damage_label = dir_info.get('damage_label', -1)
            seg_label = dir_info.get('segmentation_label', 0)
            location = dir_info['location']
            name = dir_info.get('name', '')
            type_giai_phau = dir_info.get('type_giai_phau', '')
            type_ton_thuong = dir_info.get('type_ton_thuong', '')
            hp_label = -1

            if type == 'segmentation':
                if name == "tonthuong":
                    with open(location) as f:
                        data = json.load(f)
                    if type_ton_thuong != "viem_da_day_20230620":
                        train_image_paths = data[type_ton_thuong]["train"]["images"]
                        train_mask_paths = data[type_ton_thuong]["train"]["masks"]

                        test_image_paths = data[type_ton_thuong]["test"]["images"]
                        test_mask_paths = data[type_ton_thuong]["test"]["masks"]
                        print(f"Processed {type_ton_thuong}")
                        for img_path, mask_path in zip(train_image_paths, train_mask_paths):
                            self.train_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
                        for img_path, mask_path in zip(test_image_paths, test_mask_paths):
                            self.test_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
                        print(f"{type_ton_thuong}: {len(self.train_samples) + len(self.test_samples)}")
                    else:
                        train_cnt = 0
                        for img_path, mask_path in zip(data[type_ton_thuong]["train"]["images"], data[type_ton_thuong]["train"]["masks"]):
                            for e in hp_data["train"]:
                                if e["image"] == img_path:
                                    train_cnt += 1
                                    self.train_samples.append([img_path, mask_path, position_label, damage_label, seg_label, e["label"]])
                                    break
                        test_cnt = 0
                        for img_path, mask_path in zip(data[type_ton_thuong]["test"]["images"], data[type_ton_thuong]["test"]["masks"]):
                            for e in hp_data["test_positive"]["images"]:
                                if e == img_path:
                                    test_cnt += 1
                                    self.test_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_data["test_positive"]["label"]])
                                    break
                            for e in hp_data["test_negative"]["images"]:
                                if e == img_path:
                                    test_cnt += 1
                                    self.test_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_data["test_negative"]["label"]])
                                    break
                        print(f"{type_ton_thuong}: {train_cnt}, {test_cnt}")
                elif name == "polyp":
                    with open(location) as f:
                        data = json.load(f)

                    train_image_paths = data["train"]["images"]
                    train_mask_paths = data["train"]["masks"]

                    test_image_paths = data["test"]["images"]
                    test_mask_paths = data["test"]["masks"]
                    for img_path, mask_path in zip(train_image_paths, train_mask_paths):
                        self.train_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
                    for img_path, mask_path in zip(test_image_paths, test_mask_paths):
                        self.test_samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
                    print(f"Processed {name}")
                    print(f"{name}: {len(self.train_samples) + len(self.test_samples)}")
                
            elif type == 'classification':
                with open(location) as f:
                    data = json.load(f)
                if name == "vitrigiaiphau":
                    train_image_paths = data[type_giai_phau]["train"]
                    test_image_paths = data[type_giai_phau]["test"]
                    for img_path in train_image_paths:
                        self.train_samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])
                    for img_path in test_image_paths:
                        self.test_samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])
                    print(f"Processed {type_giai_phau}")
                    print(f"{type_giai_phau}: {len(self.train_samples) + len(self.test_samples)}")
                # elif name == "hp":
                #     for e in data["train"]:
                #         self.train_samples.append([e["image"], None, position_label, damage_label, seg_label, e["label"]])
                #     for e in data["test_positive"]["images"]:
                #         self.test_samples.append([e, None, position_label, damage_label, seg_label, data["test_positive"]["label"]])
                #     for e in data["test_negative"]["images"]:
                #         self.test_samples.append([e, None, position_label, damage_label, seg_label, data["test_negative"]["label"]])
                #     print(f"Processed {name}")
        if self.mode == "train":
            # random.shuffle(self.train_samples)
            # self.samples = self.train_samples[:int(len(self.train_samples)*0.1)]
            self.samples = self.train_samples
        else:
            # random.shuffle(self.test_samples)
            # self.samples = self.test_samples[:int(len(self.test_samples)*0.1)]
            self.samples = self.test_samples

    def aug(self, image, mask):
        img_size = self.img_size
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(img_size[0], img_size[1]),])
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
                A.Resize(img_size[0], img_size[1])
            ])

        return t(image=image, mask=mask)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, position_label, damage_label, seg_label, hp_label = self.samples[idx]
        img_path = os.path.join(self.root_path, img_path)
        img = cv2.imread(img_path).astype(np.float32)

        if mask_path is not None:
            mask_path = os.path.join(self.root_path, mask_path)
            orin_mask = cv2.imread(mask_path).astype(np.float32)
        else:
            orin_mask = img

        augmented = self.aug(img, orin_mask)
        img = augmented['image']
        orin_mask = augmented['mask']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)
        orin_mask = torch.from_numpy(orin_mask.copy())
        orin_mask = orin_mask.permute(2, 0, 1)

        img /= 255.
        orin_mask = orin_mask.mean(dim=0, keepdim=True)/255.

        orin_mask[orin_mask <= 0.5] = 0
        orin_mask[orin_mask > 0.5] = 1

        if mask_path is None:
            segment_weight = 0
        else:
            segment_weight = seg_label

        return img, orin_mask, position_label, damage_label, segment_weight, hp_label

# class MultiTask(Dataset):
#     def __init__(
#             self, 
#             path="./data_dir.json", 
#             mode="train",
#             img_size=256,
#             root_path="home/s/DATA/"
#         ):
#         self.path = path
#         self.img_size = img_size
#         self.mode = mode
#         self.samples = []
#         self.root_path = root_path
#         self.load_data_from_json()

#     def load_data_from_json(self):
#         with open(self.path) as f:
#             dirs = json.load(f)["dirs"]
#         for dir in dirs:
#             name = dir["name"]
#             type = dir["type"]
#             position_label = dir.get('position_label', -1)
#             damage_label = dir.get('damage_label', -1)
#             seg_label = dir.get('segmentation_label', 0)
#             type_ton_thuong = dir.get('type_ton_thuong', "")
#             type_giai_phau = dir.get('type_giai_phau', "")
#             hp_label = -1
#             location = dir['location']
#             if type == "segmentation":
#                 if name == "polyp":
#                     with open(location) as f:
#                         data = json.load(f)
#                     image_paths = data[self.mode]["images"]
#                     mask_paths = data[self.mode]["masks"]
#                     for img_path, mask_path in zip(image_paths, mask_paths):
#                         self.samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
#                 elif name == "tonthuong":
#                     with open(location) as f:
#                         data = json.load(f)[type_ton_thuong]
#                     image_paths = data[self.mode]["images"]
#                     mask_paths = data[self.mode]["masks"]
#                     for img_path, mask_path in zip(image_paths, mask_paths):
#                         self.samples.append([img_path, mask_path, position_label, damage_label, seg_label, hp_label])
#             elif type == "classification":
#                 if name == "vitrigiaiphau":
#                     with open(location) as f:
#                         data = json.load(f)
#                     for img_path in data[type_giai_phau][self.mode]:
#                         self.samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])
#                 elif name == "hp":
#                     with open(location) as f:
#                         data = json.load(f)
#                     if self.mode == "train":
#                         image_paths = data[self.mode]
#                         for elem in image_paths:
#                             hp_label = elem["label"]
#                             img_path = elem["image"]
#                             self.samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])
#                     else:
#                         image_paths = data["test_positive"]["images"]
#                         hp_label = data["test_positive"]["label"]
#                         for img_path in image_paths:
#                             self.samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])

#                         image_paths = data["test_negative"]["images"]
#                         hp_label = data["test_positive"]["label"]
#                         for img_path in image_paths:
#                             self.samples.append([img_path, None, position_label, damage_label, seg_label, hp_label])
    
#     def aug(self, image, mask):
#         img_size = self.img_size
#         if self.mode == 'train':
#             t1 = A.Compose([A.Resize(img_size, img_size),])
#             resized = t1(image=image, mask=mask)
#             image = resized['image']
#             mask = resized['mask']
#             t = A.Compose([                
#                 A.HorizontalFlip(p=0.7),
#                 A.VerticalFlip(p=0.7),
#                 A.Rotate(interpolation=cv2.BORDER_CONSTANT, p=0.7),
#                 A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
#                 A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=0, p=0.35),
#                 A.MotionBlur(p=0.2),
#                 A.HueSaturationValue(p=0.2),                
#             ], p=0.5)

#         elif self.mode == 'test':
#             t = A.Compose([
#                 A.Resize(img_size, img_size)
#             ])

#         return t(image=image, mask=mask)
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         img_path, mask_path, position_label, damage_label, seg_label, hp_label = self.samples[index]
#         full_image_path = os.path.join(self.root_path, img_path)
#         full_mask_path = os.path.join(self.root_path, mask_path)

#         img = cv2.imread(full_image_path).astype(np.float32)
#         if mask_path is not None:
#             orin_mask = cv2.imread(full_mask_path).astype(np.float32)
#         else:
#             orin_mask = img

#         augmented = self.aug(img, orin_mask)
#         img = augmented['image']
#         orin_mask = augmented['mask']

#         img = torch.from_numpy(img.copy())
#         img = img.permute(2, 0, 1)
#         img /= 255.
        
#         orin_mask = torch.from_numpy(orin_mask.copy())
#         orin_mask = orin_mask.permute(2, 0, 1)
#         orin_mask = orin_mask.mean(dim=0, keepdim=True)/255.
#         orin_mask[orin_mask <= 0.5] = 0
#         orin_mask[orin_mask > 0.5] = 1

#         if mask_path is None:
#             segment_weight = 0
#         else:
#             segment_weight = seg_label

#         return img, orin_mask, position_label, damage_label, segment_weight, hp_label



        



if __name__ == "__main__":
    train_dataset = Multitask(img_size=384, segmentation_classes=5, mode="train")

    # # Ton thuong
    # mode = "train"
    # train_dataset = Multitask(segmentation_classes=5, mode="train", root_path="/home/s/DATA")

    # print(len(train_dataset.samples))

    # # Ton thuong
    # mode = "test"
    # ds = Multitask(path="/mnt/quanhd/DoAn/unet/dataset/test_ton_thuong_dir.json", mode=mode)
    # pprint(len(ds.samples))
    # with open("/mnt/quanhd/endoscopy/ft_ton_thuong.json") as f:  # ton thuong 
    #     data = json.load(f)
    # total = 0
    # for datum in data:
    #     d = data[datum]
    #     total += len(d[mode]["images"])
    # print(f"Ton thuong-{mode}: {total}")

    # # vi tri
    # mode = "train"
    # ds = MultiTask(path="/mnt/quanhd/DoAn/unet/dataset/test_vi_tri_dir.json", mode=mode)
    # pprint(len(ds.samples))
    # with open("/mnt/quanhd/endoscopy/vi_tri_giai_phau.json") as f: 
    #     data = json.load(f)
    # total = 0
    # for datum in data:
    #     if datum != "metadata":
    #         total += len(data[datum][mode])
    # print(f"Vi tri-{mode}: {total}")

    # # polyp
    # mode = "test"
    # ds = MultiTask(path="/mnt/quanhd/DoAn/unet/dataset/test_polyp_dir.json", mode=mode)
    # pprint(len(ds.samples))
    # with open("/mnt/quanhd/endoscopy/polyp.json") as f: 
    #     data = json.load(f)
    # print(f"Polyp-{mode}: {len(data[mode]['images'])}")

    # # hp
    # mode = "test"
    # ds = MultiTask(path="/mnt/quanhd/DoAn/unet/dataset/test_hp_dir.json", mode=mode)
    # pprint(len(ds.samples))
    # total = 0
    # with open("/mnt/quanhd/endoscopy/hp.json") as f:
    #     data = json.load(f)
    # if mode == "train":
    #     total += len(data[mode])
    # else:
    #     total += len(data["test_positive"]["images"])
    #     total += len(data["test_negative"]["images"])
    # print(f"hp-{mode}: {total}")
