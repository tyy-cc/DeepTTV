#!/bin/bash
#SBATCH --job-name=collect_Data
# ...

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python collect_data.py
