#!/bin/bash
#SBATCH --job-name=nnunet_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=00:10:00

nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs_dummyLoad Task501_MICCAI 0
