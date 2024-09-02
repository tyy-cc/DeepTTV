# Steps
Data is already included in the folder **TTV_files_ttv**, you may skip steps 1 to 5.
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
1. If using a cluster with Slurm, run **collect_data.py** by 

        sbatch collect_data.sbatch

but you need to modify this sbatch file.

2. If not using a cluster, simply run **collect_data.py**.

### 5. filter data
This step might seem to overlap with the previous one, but it gives the format of data we need for plotting. The result from step 4 is an intermediate product, which could be used for visualization and analysis.

1. If using a cluster with Slurm, run **filter_conver_to_train_corrected.py** by

        sbatch filter_conver_to_train_corrected.py

but you need to modify this sbatch file.


2. If not using a cluster, simply run **filter_conver_to_train_corrected.py**.

### Now you can see the final data in the folder <mark>TTV_files_ttv</mark>. For the complete dataset used in this project is not included in this repository due to the size. If you would like to use it, please contact the author.
