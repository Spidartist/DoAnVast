python mainv2.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-4 --max_lr 1e-6 --img_size 384 --gpu "1" --batch_size 4 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
