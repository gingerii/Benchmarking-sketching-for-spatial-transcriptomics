#!/bin/bash

# Define the values for each parameter
fractions=(0.6 0.7 0.8 0.9)
#(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#seeds=(0 1 2 3 4 5 6 7 8 9)
#knns=(5 10 20)
#betas=(0 1 2)

# # Iterate over each combination of parameters
# for fraction in "${fractions[@]}"; do
    
#     echo "Submitting job with fraction=$fraction"
#     sbatch slurm_script.sh "$fraction"
        
    
# done
#for seed in "${seeds[@]}"; do

for fraction in "${fractions[@]}"; do
    
    #for beta in "${betas[@]}"; do
    echo "Submitting job with fraction=$fraction"
    sbatch slurm_script.sh "$fraction"
done
    
#done