#!/bin/bash
#SBATCH --job-name=generate_data_files
# ...
cd $SLURM_SUBMIT_DIR
module load anaconda3

python file_generator_zip.py $1 
