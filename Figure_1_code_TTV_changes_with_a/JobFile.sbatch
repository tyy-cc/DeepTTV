#!/bin/bash
#SBATCH --job-name=this
#SBATCH --account=hive-gli323
#SBATCH -p hive-nvme-sas
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --time=5:00:00 
#SBATCH --mem-per-cpu=10G 
#SBATCH --output=kepler88_obs_2016_para2013_7_19_out/kepler88_progress_%j.out

cd $SLURM_SUBMIT_DIR
module load anaconda3

python file_generator_zip.py $1 
# python plot_and_compare.py $1
# python draft.py $1