#!/bin/bash

# Number of parallel processes
max_processes=2

# Define the task options
tasks=("S_D")
smto_options=("imtl" "baseline")
lr_options=(0.005 0.001 0.0005 0.0001)
wd_options=(1e-4 1e-5 0.0)

# Initialize a semaphore
semaphore=0


# Loop over the task options
for smto in "${smto_options[@]}"; do
    for task in "${tasks[@]}"; do
        for lr in "${lr_options[@]}"; do
            for wd in "${wd_options[@]}"; do
                # Check the semaphore count
                if [ "$semaphore" -ge "$max_processes" ]; then
                    wait
                    semaphore=0
                fi
                
                # Execute the command in the background
                CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py \
                    --optimizer "$smto" \
                    --dataset cityscapes \
                    --model resnet50 \
                    --shape 256 128 \
                    --aug \
                    --n_classes 7 \
                    --tasks "$task" \
                    --lr "$lr" \
                    --p 0 \
                    --weight_decay "$wd" \
                    --batch_size 32 \
                    --decay_lr \
                    --num_epochs 150 \
                    --analysis_test "grad_metrics" \
                    --n_runs 1 &

                # Increment the semaphore count
                ((semaphore++))
            done
        done
    done
done

# Wait for all background processes to finish
wait