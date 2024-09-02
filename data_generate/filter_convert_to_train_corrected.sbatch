#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --account=hive-gli323
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --time=10:00  
#SBATCH --mem-per-cpu=80G 
#SBATCH --output=filter_convert_to_train_corrected_%j.out

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python filter_convert_to_train_corrected.py
