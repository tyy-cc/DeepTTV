#!/bin/bash
#SBATCH --job-name=this
#SBATCH --account=hive-gli323
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --time=1:00:00 
#SBATCH --mem-per-cpu=10G 
#SBATCH --output=run.out

cd $SLURM_SUBMIT_DIR


./bin/simulate sample/Test_mass



