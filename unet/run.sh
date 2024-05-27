python main.py --type_pretrained endoscopy3 --task classification --type_cls HP --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "0" --type_opt "Adam"
python main.py --type_pretrained endoscopy3 --task classification --type_cls HP --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" --img_size 336 --gpu "0" --type_opt "SGD"

