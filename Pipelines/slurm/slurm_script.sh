#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --ntasks=1
#SBATCH --mem=1M 
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#Skipping many options! see man sbatch
# From here on, we can start our program

python3 test.py 