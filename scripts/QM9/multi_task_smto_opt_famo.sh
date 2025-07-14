#!/bin/bash

# Number of parallel processes
max_processes=1

# Define the task options
tasks=("mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv")
smto_options=("famo")
lr_options=(0.005 0.001 0.0005 0.0001)
gamma_options=(0.01 0.001 0.0001)
p_options=(0.0)

# Initialize a semaphore
semaphore=0

# Loop over the task options
for smto in "${smto_options[@]}"; do
    for task in "${tasks[@]}"; do
        for lr in "${lr_options[@]}"; do
            for gamma in "${gamma_options[@]}"; do
                for p in "${p_options[@]}"; do
                    # Check the semaphore count
                    if [ "$semaphore" -ge "$max_processes" ]; then
                        wait
                        semaphore=0
                    fi
                    
                    # Execute the command in the background
                    CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py \
                        --optimizer "$smto" \
                        --dataset QM9 \
                        --model mpnn \
                        --tasks "$task" \
                        --lr "$lr" \
                        --p "$p" \
                        --weight_decay 0 \
                        --decay_lr \
                        --batch_size 120 \
                        --num_epochs 300 \
                        --gamma "$gamma" \
                        --n_runs 1 &
                    
                    # Increment the semaphore count
                    ((semaphore++))
                
                done
            done
        done
    done
done

# Wait for all background processes to finish
wait