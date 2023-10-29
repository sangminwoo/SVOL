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
sketch_dataset=quickdraw  # sketchy / tu_berlin / quickdraw
eval_bs=1
num_layers=2
num_frames=32
num_queries_per_frame=10
num_queries=$(($num_frames*$num_queries_per_frame))
set_cost_bbox=5
set_cost_giou=1
set_cost_class=2
sketch_head=svanet  # svanet / sketch_detr
backbone=resnet  # vit / resnet
# mode=concat_to_qry  # concat_to_seq / append_to_seq / concat_to_qry
matcher=per_frame_matcher  # per_frame_matcher / video_matcher
resume=/mnt/server15_hard2/sangmin/code/svol/save/best_model_\
${video_dataset}_${sketch_dataset}_${sketch_head}_${backbone}_\
${num_layers}l_${num_frames}f_${num_queries}q_\
${set_cost_bbox}_${set_cost_giou}_${set_cost_class}.ckpt
gpu="6"
port=23000

CUDA_VISIBLE_DEVICES=${gpu} \
torchrun \
--master_port=$port \
--nproc_per_node=1 test.py \
--root ${root} \
--anno_root ${anno_root} \
--video_dataset ${video_dataset} \
--sketch_dataset ${sketch_dataset} \
--eval_bs ${eval_bs} \
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
--resume ${resume}
# --use_neptune
# --mode ${mode} \