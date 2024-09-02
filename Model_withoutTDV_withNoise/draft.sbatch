#!/bin/bash
#SBATCH --job-name=file_generate
#SBATCH --account=bboy-delta-cpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00
#SBATCH --partition=cpu 
#SBATCH --output=file_generator.output

module reset
module load anaconda3_cpu
#pip install pprintpp
python ./draft.py