#!/bin/bash
#SBATCH --job-name=this
# ...

cd $SLURM_SUBMIT_DIR
module load anaconda3

python file_generator_zip.py $1 