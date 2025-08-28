#!/bin/bash -l
# How long should I job run for (note that right now 10 min is the cutoff)
#SBATCH --account=frostlab
#SBATCH --time=02:00:00 
# Number of CPU cores, in this case 1 core, because this command does not support multithreading
#SBATCH --ntasks-per-node=15
# Number of compute nodes to use
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10GB
# Name of the output files to be created. If not specified the outputs will be joined
#SBATCH --output=/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/HPC/outs/%x.%j.out
#SBATCH --error=/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/HPC/outs/%x.%j.err

#Activate your conda environment
conda activate r-kernel

# The code you want to run in your job
python rasp_distance_driver.py --knn "$1" --beta "$2" --fraction "$3" --seed "$4"