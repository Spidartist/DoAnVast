# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged ung_thu_thuc_quan_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_da_day_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_thuc_quan_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_loet_hoanh_ta_trang_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"

# python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 448 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" 


python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy4 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy3 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
# python main.py --type_damaged ung_thu_thuc_quan_20230620 --type_pretrained endoscopy4 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
# python main.py --type_damaged viem_da_day_20230620 --type_pretrained endoscopy4 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
# python main.py --type_damaged viem_loet_hoanh_ta_trang_20230620 --type_pretrained endoscopy4 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
# python main.py --type_damaged viem_thuc_quan_20230620 --type_pretrained endoscopy4 --task segmentation --root_path "/home/s/DATA/" --type_seg "TonThuong" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"


# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy2 --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 0 --lr 1e-3
# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy2 --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 10 --lr 1e-3
# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 10 --lr 1e-4

# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 100

# python main.py --type_pretrained endoscopy --task classification --type_cls HP --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 256
# python main.py --type_pretrained endoscopy --task classification --type_cls HP --num_freeze 0 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 256
# python main.py --type_pretrained endoscopy3 --task classification --type_cls vitri --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "1" --type_opt "SGD"
# python main.py --type_pretrained endoscopy3 --task classification --type_cls vitri --num_freeze 0 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "1" --type_opt "SGD"

# python main.py --type_pretrained endoscopy3 --task classification --type_cls vitri --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "1" --type_opt "Adam"
# python main.py --type_pretrained endoscopy3 --task classification --type_cls vitri --num_freeze 0 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "1" --type_opt "Adam"

# python main.py --type_pretrained endoscopy --task segmentation --type_seg benchmark --num_freeze 60 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW"
# python main.py --type_pretrained endoscopy1 --task segmentation --type_seg benchmark --num_freeze 60 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW"
# python main.py --type_pretrained endoscopy2 --task segmentation --type_seg benchmark --num_freeze 00 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW"

# python main.py --type_pretrained endoscopy --task segmentation --type_seg benchmark --num_freeze 60 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "encoder"
# python main.py --type_pretrained endoscopy --task segmentation --type_seg benchmark --num_freeze 60 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"

# python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-4 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "SGD" --type_encoder "target_encoder"
# python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-4 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "SGD" --type_encoder "target_encoder"

