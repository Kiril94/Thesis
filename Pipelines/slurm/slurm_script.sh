#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --ntasks=1
# number of cpus we want to allocate for each program
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=1000M
#SBATCH -p gpu --gres=gpu:gtx1080:2
#SBATCH --time=00:20:00
# Skipping many options! see man sbatch
# From here on, we can start our program
echo $CUDA_VISIBLE_DEVICES
python3 plan_preprocess.py