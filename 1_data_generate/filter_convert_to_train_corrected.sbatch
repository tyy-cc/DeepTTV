#!/bin/bash
#SBATCH --job-name=filter
# ...
cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python filter_convert_to_train_corrected.py
