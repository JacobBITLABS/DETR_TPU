# python3 prep_train.py

python3 main.py \
  --device tpu  \
  --dataset_file "coco" \
  --coco_path ../wilder_dataset/ \
  --output_dir /home/jacobnielsen/DETR_run/DETR_TPU/output \
  --resume 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth' \
  --lr 1e-5 \
  --lr_backbone 1e-6 \
  --epochs 3