python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-4 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "SGD" --type_encoder "encoder"
python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 0 --max_lr 1e-6 --min_lr 2e-4 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "SGD" --type_encoder "target_encoder"

# python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 6 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW"
# python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 0 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 2 --type_opt "AdamW"


