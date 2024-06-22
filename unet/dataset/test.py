import os


path = "/mnt/tuyenld/data/endoscopy/public_dataset/TrainDataset/masks"

dirs = os.listdir(path)
cnt = 0
for dir in dirs:
    if "c" in dir:
        cnt += 1
print(len(dirs))
print(cnt)