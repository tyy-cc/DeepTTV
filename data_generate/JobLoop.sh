for i in {0..450}; do
  sbatch JobFile.sbatch $i
done