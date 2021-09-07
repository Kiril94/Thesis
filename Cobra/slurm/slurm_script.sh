#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --ntasks=1
#SBATCH --mem=1M 
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=00:05:00
#Skipping many options! see man sbatch
# From here on, we can start our program

python3 test_slurm.py 