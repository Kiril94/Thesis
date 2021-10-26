#!/bin/bash
#SBATCH --job-name=nnunet_help
#SBATCH --partition=image1
# number of cpus we want to allocate for each program
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=00:01:00
# Skipping many options! see man sbatch
# From here on, we can start our program
nnUNet_train -h 
