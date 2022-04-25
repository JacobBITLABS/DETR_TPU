python3 prep_train.py

python3 main.py \
  --dataset_file "coco" \
  --coco_path ../DroneVis_DETR/ \
  --output_dir /home/jacobnielsen/DETR_run/DETR_TPU/output \
  --resume "detr-r50_no-class-head.pth" \
  --lr 1e-5 \
  --lr_backbone 1e-6 \
  --epochs 2