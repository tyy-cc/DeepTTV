#!/bin/bash
#SBATCH --job-name=do_TTV
# ...

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python collect_data.py
