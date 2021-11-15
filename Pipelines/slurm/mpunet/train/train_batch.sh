#!/bin/bash
#SBATCH --job-name=mpunet
# number of cpus we want to allocate for each program
##SBATCH -N 1
##SBATCH --ntasks-per-node=24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=10:00:00
##SBATCH –-output=/home/fjn197/Thesis/Pipelines/slurm/out.out
# Skipping many options! see man sbatch
# From here on, we can start our program
mp train --project_dir /home/fjn197/Thesis/Pipelines/data/mpunet/MICCAI_project --epochs 200 --no_images
