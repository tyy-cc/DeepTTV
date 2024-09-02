#!/bin/bash
#SBATCH --job-name=do_TTV
#SBATCH --account=hive-gli323
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=100G 
#SBATCH --output=plot_compare_%j.output

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python plot_and_compare.py
