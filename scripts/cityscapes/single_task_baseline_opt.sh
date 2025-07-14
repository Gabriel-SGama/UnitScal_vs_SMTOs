#!/bin/bash

# Number of parallel processes
max_processes=2

# Define the task options
tasks=("S" "D" "I")
lr_options=(0.005 0.001 0.0005 0.0001)
p_options=(0.0)
wd_options=(1e-4 1e-5 0.0)

# Initialize a semaphore
semaphore=0

# Loop over the task options
for task in "${tasks[@]}"; do
    for lr in "${lr_options[@]}"; do
        for p in "${p_options[@]}"; do
            for wd in "${wd_options[@]}"; do
                # Check the semaphore count
                if [ "$semaphore" -ge "$max_processes" ]; then
                    wait
                    semaphore=0
                fi

                CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py \
                --optimizer baseline \
                --dataset cityscapes \
                --model resnet18 \
                --shape 512 256 \
                --aug \
                --n_classes 19 \
                --tasks "$task" \
                --lr "$lr" \
                --p "$p" \
                --weight_decay "$wd" \
                --batch_size 32 \
                --decay_lr \
                --num_epochs 150 \
                --n_runs 1 &

                # Increment the semaphore count
                ((semaphore++))
            done
        done
    done
done

wait
