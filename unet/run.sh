python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 448 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" 
python main.py --type_pretrained im1k --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-5 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 448 --gpu "1" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder" 

# python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 0.0 --min_lr 3e-5 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"
# python main.py --type_pretrained endoscopy4 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 0.0 --min_lr 3e-5 --lr 1e-3--root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW" --type_encoder "target_encoder"


# python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 6 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW"
# python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 0 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW"


