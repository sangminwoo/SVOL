#!/bin/bash

DATASET="sketchy tu_berlin quickdraw"
gpu=0

for dataset in $DATASET
do
	CUDA_VISIBLE_DEVICES=${gpu} \
	python sketch_vit_feature_extractor.py --dataset $dataset
done

for dataset in $DATASET
do
CUDA_VISIBLE_DEVICES=${gpu} \
	python sketch_vit_finetune.py --dataset $dataset
done

# CUDA_VISIBLE_DEVICES=0 python sketch_vit_feature_extractor.py --dataset sketchy
# CUDA_VISIBLE_DEVICES=1 python sketch_vit_feature_extractor.py --dataset tu_berlin
# CUDA_VISIBLE_DEVICES=2 python sketch_vit_feature_extractor.py --dataset quickdraw
# CUDA_VISIBLE_DEVICES=3 python sketch_vit_finetune.py --dataset sketchy
# CUDA_VISIBLE_DEVICES=4 python sketch_vit_finetune.py --dataset tu_berlin
# CUDA_VISIBLE_DEVICES=5 python sketch_vit_finetune.py --dataset quickdraw