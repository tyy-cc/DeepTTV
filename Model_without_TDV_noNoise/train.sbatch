#!/bin/bash
#SBATCH --job-name=64_0.0008
#SBATCH --account=gts-gli323-cygnus
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=32:00:00  
#SBATCH --output=test_pred_%j.output

cd $SLURM_SUBMIT_DIR

module load anaconda3

conda activate forDLa100
# srun python test_pred.py
srun python train.py