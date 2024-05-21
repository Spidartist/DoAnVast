# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged ung_thu_thuc_quan_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_da_day_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_thuc_quan_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_loet_hoanh_ta_trang_20230620 --type_pretrained im1k --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"

# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged ung_thu_thuc_quan_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_loet_hoanh_ta_trang_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"
# python main.py --type_damaged viem_thuc_quan_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong"


# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy2 --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 0 --lr 1e-3
# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy2 --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 10 --lr 1e-3
# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 10 --lr 1e-4

# python main.py --type_damaged ung_thu_da_day_20230620 --type_pretrained endoscopy --root_path "/mnt/tuyenld/data/endoscopy/" --type_seg "TonThuong" --num_freeze 100

python main.py --type_pretrained endoscopy1 --task classification --type_cls HP --num_freeze 0 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" 
python main.py --type_pretrained endoscopy1 --task classification --type_cls HP --num_freeze 10 --lr 1e-3 --root_path "/mnt/tuyenld/data/endoscopy/" 

