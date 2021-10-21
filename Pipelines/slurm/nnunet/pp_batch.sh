#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
# number of cpus we want to allocate for each program
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=00:20:00
##SBATCH –-output=/home/fjn197/Thesis/Pipelines/slurm/out.out
# Skipping many options! see man sbatch
# From here on, we can start our program
python3 /home/fjn197/Thesis/Pipelines/nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 500 --verify_dataset_integrity