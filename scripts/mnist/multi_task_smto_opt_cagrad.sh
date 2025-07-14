#!/bin/bash

# Number of parallel processes
max_processes=4

# Define the task options
tasks=("CL_RR" "CL_CR" "RL_RR")
smto_options=("cagrad")
lr_options=(0.01 0.075 0.005 0.0025 0.001 0.00075 0.0005)
c_options=(0.2 0.5 0.8)
p_options=(0.0 0.1 0.2 0.3 0.4 0.5)

# Initialize a semaphore
semaphore=0

# Loop over the task options
for smto in "${smto_options[@]}"; do
    for task in "${tasks[@]}"; do
        for lr in "${lr_options[@]}"; do
            for c in "${c_options[@]}"; do
                for p in "${p_options[@]}"; do
                    # Check the semaphore count
                    if [ "$semaphore" -ge "$max_processes" ]; then
                        wait
                        semaphore=0
                    fi
                    
                    # Execute the command in the background
                    CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py \
                        --optimizer "$smto" \
                        --dataset mnist \
                        --model lenet \
                        --tasks "$task" \
                        --lr "$lr" \
                        --p "$p" \
                        --weight_decay 0 \
                        --decay_lr \
                        --num_epochs 100 \
                        --conv_rate "$c" \
                        --n_runs 5 &
                    
                    # Increment the semaphore count
                    ((semaphore++))
                
                done
            done
        done
    done
done

# Wait for all background processes to finish
wait
