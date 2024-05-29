python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 10 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 8
python main.py --type_pretrained endoscopy3 --task segmentation --type_seg benchmark --num_freeze 0 --lr 1e-3 --root_path "/home/s/DATA/" --img_size 512 --gpu "0" --batch_size 8

