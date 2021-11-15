#!/bin/bash
#SBATCH --job-name=nnunet_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=00:15:00

nnUNet_train 2d nnUNetTrainerV2_5epochs Task501_MICCAI 0
