#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./logs/%j_out.txt
#SBATCH -e ./logs/%j_err.txt
#SBATCH --gres=gpu:1

root=/mnt/server15_hard2/sangmin/data/svol/
anno_root=/mnt/server15_hard2/sangmin/data/svol/annos/
video_dataset=imagenet_vid
sketch_dataset=quickdraw
num_iters=100000
lr_drop_step=30000
log_interval=100
val_interval=2000
early_stop_patience=10
bs=16
lr=1e-4
num_layers=2
num_frames=32
num_queries_per_frame=10
num_queries=$(($num_frames*$num_queries_per_frame))
set_cost_bbox=5
set_cost_giou=1
set_cost_class=2
sketch_head=svanet  # svanet / sketch_detr
backbone=resnet  # vit / resnet / s3d
# mode=concat_to_qry  # concat_to_seq / append_to_seq / concat_to_qry
matcher=per_frame_matcher  # per_frame_matcher / video_matcher
gpu="6,7"
port=24000

CUDA_VISIBLE_DEVICES=${gpu} \
torchrun \
--master_port=$port \
--nproc_per_node=2 train.py \
--root ${root} \
--anno_root ${anno_root} \
--video_dataset ${video_dataset} \
--sketch_dataset ${sketch_dataset} \
--num_iters ${num_iters} \
--lr_drop_step ${lr_drop_step} \
--log_interval ${log_interval} \
--val_interval ${val_interval} \
--bs ${bs} \
--lr ${lr} \
--num_layers ${num_layers} \
--num_frames ${num_frames} \
--num_queries ${num_queries} \
--num_queries_per_frame ${num_queries_per_frame} \
--set_cost_bbox ${set_cost_bbox} \
--set_cost_giou ${set_cost_giou} \
--set_cost_class ${set_cost_class} \
--sketch_head ${sketch_head} \
--backbone ${backbone} \
--matcher ${matcher} \
--use_neptune 
# --zeroshot_category_eval
# --zeroshot_dataset_eval
# --tight_frame_sampling
# --mode ${mode} \
# --eval_untrained