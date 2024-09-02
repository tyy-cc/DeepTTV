#!/bin/bash
#SBATCH --job-name=plot_result
# ...

cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python plot_figure1.py
