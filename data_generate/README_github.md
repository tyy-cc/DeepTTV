# Steps
### 1. compile Grit: 
       cd Template
       cd src
       make
### 2. Parameters and other settings can be changed in **file_generator_zip.py**. 


### 3. Submit jobs
1. If using a cluster with Slurm, you may use **JobLoop.sh** to submit jobs, number in for loop defines the number of jobs you would like to submit. **JobFile.sbatch** should be modified.

        chmod +x JobLoop.sh
        ./JobLoop.sh

2. If not using a cluster, simply run **file_generator_zip.py**.

### 4. collect data
1. If using a cluster with Slurm, run **plot_and_compare.py** by 

        sbatch plot_and_compare.sbatch

but you need to modify this sbatch file.

#### 5. filter data
1. If using a cluster with Slurm, run **filter_conver_to_train_corrected.py** by

        sbatch filter_conver_to_train_corrected.py

but you need to modify this sbatch file.

### Now you can see the final data in the folder <mark>TTV_files_ttv</mark>.
