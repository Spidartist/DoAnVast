python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-4 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 8 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
python main.py --type_pretrained endoscopy_mae --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-3 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 8 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0

python main.py --type_pretrained endoscopy_mae_final --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-4 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 8 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
python main.py --type_pretrained endoscopy_mae_final --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-3 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 8 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0


python main.py --type_pretrained mae --task sesgmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-3 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 4 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
python main.py --type_pretrained endoscopy3 --task segmentation --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "benchmark" --num_freeze 0 --min_lr 2e-5 --lr 1e-3 --max_lr 1e-6 --img_size 384 --gpu "0" --batch_size 4 --type_opt "AdamW" --type_encoder "target_encoder" --scale_lr 0
