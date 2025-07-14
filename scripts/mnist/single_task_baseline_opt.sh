#!/bin/bash

# Define the task options
tasks=("CL" "RR" "CR" "RL")
lr_options=(0.01 0.075 0.005 0.0025 0.001 0.00075 0.0005)
p_options=(0.0 0.1 0.2 0.3 0.4 0.5)

# Loop over the task options
for task in "${tasks[@]}"; do
    for lr in "${lr_options[@]}"; do
        for p in "${p_options[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py \
                --optimizer baseline \
                --dataset mnist \
                --model lenet \
                --tasks "$task" \
                --lr "$lr" \
                --p "$p" \
                --weight_decay 0 \
                --decay_lr \
                --num_epochs 100 \
                --n_runs 5
        done
    done
done
