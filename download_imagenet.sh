#!/usr/bin/env bash
# Download ImageNet ILSVRC 2012 data for benchmarking attribution methods.

cd data/datasets/imagenet

# (download datasets from torrent files)
# put tar inside dataset folder
cp /download/path/ILSVRC2012_img_train.tar
cp /download/path/ILSVRC2012_img_val.tar

# clone this repository
git clone https://gitlab.com/nicolalandro/download_and_prepare_imagenet_dataset.git 

# copy files
cp download_and_prepare_imagenet_dataset/*.sh .
cp download_and_prepare_imagenet_dataset/*.py .
cp download_and_prepare_imagenet_dataset/*.json .

# untar
./untar_train.sh
./untar_val.sh

# rename with human class names
python3.6 id_to_class_train.py
python3.6 id_to_class_val.py


