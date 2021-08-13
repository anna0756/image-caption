#!/bin/bash

echo "[INFO] Starting..."

# train
python train_image_folder.py --data_format image_folder --arch resnet18 \
--batch-size 512 --workers 2 --classes 3 --epoch 20 --in-shape 3 224 224 \
--data-path ./dataset/image_folder/

# test
#python train_image_folder.py --data_format image_folder --arch resnet18 \
#--batch-size 512 --workers 2 --classes 3 --epoch 20 --in-shape 3 224 224 \
#--data-path ./dataset/image_folder/

echo "[INFO] Done."
