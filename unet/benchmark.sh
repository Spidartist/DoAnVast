# python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --img_size 512 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" --train_ratio 0.5
# python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-3 --max_lr 1e-6 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-4 --max_lr 1e-6 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-6 --lr 1e-5 --max_lr 1e-6 --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
