#!/bin/bash
#SBATCH --job-name=do_TTV
#SBATCH --account=hive-gli323
#SBATCH -p hive-nvme
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10G 
#SBATCH --output=draft_%j.output

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python draft.py
