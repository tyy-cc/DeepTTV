for i in {1..10}; do
  sbatch JobFile.sbatch $i
done