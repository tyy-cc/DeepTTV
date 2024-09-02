# Steps
### 1. compile Grit: 
       cd Template
       cd src
       make
### 2. change semi-major axis and other settings
1. open **file_generator_zip.py** 
2. locate and change **a_transit** and **a_nontransit**
3. **num_test** defines the number of runs for each job, but note that more runs requires more walltime which could make job pending for a long time.

### 3. submit jobs
1. use **JobLoop.sh** to submit jobs, number in for loop defines the number of jobs you would like to submit.

        chmod +x JobLoop.sh
        ./JobLoop.sh

### 4. collect data
1. run **plot_and_compare.py** by 

        sbatch plot_and_compare.sbatch

#### 5. filter data
1. run **filter_conver_to_train_corrected.py** by

        sbatch filter_conver_to_train_corrected.py

### Now you can see the final data in the folder <mark>TTV_files_ttv</mark>.

# Code Logistics
### 1. Grit
Under **Template/src/system.cpp** 
#### 1.1 
 If planet current distance from the star is 10 times larger than the initial state, stop and show "Planet ejected".

#### 1.2 
If values of position are Nan, stop and show "Nan detected".

#### 1.3 
Within a certain range (save computation time), check if transit happens.
##### 1.3.1 
If non-transit transits, stop and show "nontransit transit".
##### 1.3.2 
Comapre transit duration and transit interval with the observation. The interval should vary within 0.6% and the duration difference should below one hour.  