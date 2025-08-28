#!/bin/bash

# Define the values for each parameter
fractions=(0.7 0.8 0.9 0.99)
seeds=(0 1 2 3 4 5)

# Iterate over each combination of parameters
for fraction in "${fractions[@]}"; do
    for seed in "${seeds[@]}"; do
        
        echo "Submitting job with fraction=$fraction, seed=$seed"
        sbatch slurm_script.sh "$fraction" "$seed"
       
    done
done