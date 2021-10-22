#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
# number of cpus we want to allocate for each program
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=00:10:00
##SBATCH –-output=/home/fjn197/Thesis/Pipelines/slurm/out.out
# Skipping many options! see man sbatch
# From here on, we can start our program
nnUNet_predict -i /home/fjn197/Thesis/Pipelines/data/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Test/imagesTs -o /home/fjn197/Thesis/Pipelines/data/nnunet/nnUNet_predictions -t 500 
