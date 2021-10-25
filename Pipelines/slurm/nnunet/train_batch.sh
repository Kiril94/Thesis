#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
# number of cpus we want to allocate for each program
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=24:00:00
##SBATCH –-output=/home/fjn197/Thesis/Pipelines/slurm/out.out
# Skipping many options! see man sbatch
# From here on, we can start our program
nnUNet_train 2d nnUNetTrainerV2 Task500_Test 0 -c
