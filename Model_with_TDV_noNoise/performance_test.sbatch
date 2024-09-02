#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mem=50g
#SBATCH --account=bboy-delta-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --partition=gpuA100x4 
#SBATCH --output=perfor_test_%j.output
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

module purge
python ./performance_test_set.py
