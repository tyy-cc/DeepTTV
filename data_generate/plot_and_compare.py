import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os.path
from os import path
import json
import pickle
from utils import get_ttv_duration
import pprint
from zipfile import ZipFile
import io
import glob 
import sys 

cwd = os.getcwd()
cwd = str(cwd)
Folder_name = "kepler88_obs_2016_para2013_3_12"

if not path.exists(cwd + '/TTV_files_ttv'):
    os.mkdir(cwd + '/TTV_files_ttv')

plot_path =  cwd + '/TTV_files_ttv'


parameter_names = ['m_t', 'm_nont', 
                   'e_t', 'e_nont', 
                   'incli_t', 'incli_nont', 
                   'a_t', 'a_nont', 
                   'omega_t', 'omega_nont',
                   'Omega_t', 'Omega_nont', 
                   'Mean_ano_t', 'Mean_ano_nont']


count = 0
par_count = 0 

total_dic = {}

files = glob.glob(cwd + "/sim_folder_"+ Folder_name +"/num_*")


for file in files:
    cur_parent_path = file

    if not path.exists(cur_parent_path):
        print(cur_parent_path + ' not exists!')
        exit()

    print(f'currently reading {file}')

    with open(cur_parent_path + '/file_names.txt', "r") as file_names: # open file_names in num_* folder

        with ZipFile(cur_parent_path + '/reuslt.zip', 'r') as zip:
            for j, name in enumerate(file_names): # iterate Test
                
                
                name = name.split() # num_*Test_*, *file name
                
                TTV_file_name = name[0] + '.txt' # e.g., num_*_Test_*.txt

                ''' 
                1. check if the file name is valided/completed, and the file is in [*, 3] shape,
                ''' 
                try:
                    with io.BufferedReader(zip.open(TTV_file_name, mode='r')) as TTV_file:
                        transit_data = np.genfromtxt(TTV_file, dtype='str').reshape(-1, 3) 
                except:
                    continue 
                    

                if len(transit_data) == 0 or len(transit_data) != 135*3:
                    print(TTV_file_name + ' not completed')
                    continue 

                Test_dic = {}
                
                labels = name[1].split('_')
                flag = 1
                labels_number = []
                for label in labels:
                    if flag == 1:
                        flag *= -1
                        continue 
                    else:
                        labels_number.append(float(label))
                        flag *= -1
                parameters_dic = {}
                for k, para in enumerate(labels_number):
                    parameters_dic[parameter_names[k]] = para
                
                TTV_list, duration_list, \
                transit_list, fitted_Period_days, \
                percent_tdv, tdv_list = get_ttv_duration(transit_data)

                parameters_dic['P_t'] = fitted_Period_days


                Test_dic['TTV_list'] = TTV_list # wrt 134
                Test_dic['duration_list'] = duration_list
                Test_dic['transit_list'] = transit_list 
                Test_dic['parameters'] = parameters_dic 
                Test_dic['percent_tdv'] = percent_tdv
                Test_dic['name'] = name[0]\
                count += 1

                with open(cwd + '/TTV_files_ttv/kepler88_obs_2016_para2013_3_12_2013parm.pkl', 'ab') as pkl_file:
                    pickle.dump(Test_dic, pkl_file)

print(len(files), count)

