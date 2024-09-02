
import os
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import json
import pickle
import pprint
from utils import *

cwd = os.getcwd()
cwd = str(cwd)

TTV_lower = 0.5 # days
TTV_upper = 1.5

TDV_lower = 0.15 # hours
TDV_upper = 0.6
TDV_max = 0.5
TDV_min = 0.002
CONSIDER_TDV = True 
CONSIDER_TTV = False

# parameters in dic['parameters']
parameter_names = ['m_t', 'm_nont', 
                   'e_t', 'e_nont', 
                   'incli_t', 'incli_nont', 
                   'a_t', 'a_nont', 
                   'omega_t', 'omega_nont',
                   'Omega_t', 'Omega_nont', 
                   'Mean_ano_t', 'Mean_ano_nont', 
                   'P_t'] # note P_t here is fitted_Period_days

# labels we want to train the model
label_names = ['m_t', 'm_nont', 
               'e_t', 'e_nont', 
               'incli_t', 'incli_nont', 
               'omega_t', 'omega_nont', 
               'Omega_t', 'Omega_nont', 
               'Mean_ano_t', 'Mean_ano_nont', 
               'a_nont']
ttvs = []
tdvs = []
per_tdv = []
transit_times = []
durations = []
labels = []
dur_coefs = []
dur_intercepts = []
P_dur_info = []
file_names = []
total_cnt = 0
cnt = 0

# read data
with open(cwd + '/TTV_files_ttv/kepler88_obs_2016_para2013_3_12_2013parm.pkl', 'rb') as fileo:
    try:
        while True:
            data_dic = pickle.load(fileo)
            total_cnt += 1

            ttv_temp = data_dic['TTV_list']
            file_name = data_dic['name']
            fitted_P_temp = data_dic['parameters']['P_t']
            duration_temp = data_dic['duration_list']
            average_dur_temp = np.mean(duration_temp)
            per_tdv_temp = (duration_temp - average_dur_temp)/average_dur_temp

            max_ttv = np.max(np.abs(ttv_temp))
            max_tdv = np.max(np.abs(per_tdv_temp))
            ttv_amplitude = (np.max(ttv_temp) - np.min(ttv_temp)) * 24 *60 *60
            tdv_amplitude = np.max(per_tdv_temp) - np.min(per_tdv_temp)

            if CONSIDER_TDV and CONSIDER_TTV:
                condition = ttv_amplitude > 2 and  max_tdv < TDV_max and tdv_amplitude > TDV_min
            elif CONSIDER_TDV:
                condition = max_tdv < TDV_max
            else:
                condition = True

            if condition: 
                ttvs.append(ttv_temp)
                per_tdv.append(per_tdv_temp)
                durations.append(data_dic['duration_list'])
                transit_times.append(data_dic['transit_list'])
                P_dur_info.append(fitted_P_temp)
                file_names.append(file_name)

                temp_label = []

                for label in label_names:
                    temp_label.append(data_dic['parameters'][label])
                labels.append(temp_label)

                cnt += 1 


    except EOFError:
        pass

print(f'Before filtering, there are {total_cnt} data, {cnt} data left after filtering. ({TTV_lower} < max_ttv < {TTV_upper}, tdv filter {CONSIDER_TDV}.')


data = []
data.append(np.array(ttvs))
data.append(np.array(per_tdv))
data.append(np.array(durations))
data.append(np.array(transit_times))
data.append(np.array(P_dur_info))
data.append(np.array(labels))


data_filtered = data


file_filtered = open(cwd + '/TTV_files_ttv/kepler88_obs_2016_para2013_Data_3_12_tdv_0.5_average_corrected.pkl', 'wb')

pickle.dump(data_filtered, file_filtered)
file_filtered.close()

