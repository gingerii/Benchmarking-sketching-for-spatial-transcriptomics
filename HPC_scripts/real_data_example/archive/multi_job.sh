#!/bin/bash

# Define the values for each parameter
fractions=(0.5 0.6 0.7 0.8 0.9 0.99)
knns=(1 20 50 60 70)
betas=(0 1 2)
seeds=(0 1 2 3 4)

# Iterate over each combination of parameters
for fraction in "${fractions[@]}"; do
    for knn in "${knns[@]}"; do
        for beta in "${betas[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Submitting job with knn=$knn, beta=$beta, fraction=$fraction, seed=$seed"
                sbatch slurm_script2.sh "$knn" "$beta" "$fraction" "$seed"
            done
        done
    done
done