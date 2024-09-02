import torch
import torch.nn as nn
import numpy as np
import time
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
import re
import os
import pandas as pd
import pickle
from train_utils import * #get_train_data, get_tensor_training_data, train_and_eval, print_model_para, predict, create_mask, predict_from_obs, predict_from_obs_inter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys 
from torchsummary import summary
import time
from sklearn.preprocessing import MinMaxScaler
import transformers as tf
from timeit import default_timer as timer
plt.rcParams.update({'font.size': 14})

#### training data ######
INCLUDE_APM = False
two_input = True
Earth_in_Solar = 3.0027e-6 # mass
Jupiter_in_Solar = 9.547919e-4 # mass

PLOT_FIG = "hist" #"scatter" # "mad"
SCALE_INPUT = 'default'
SCALE_OUTPUT = 'MinMax01'

LABEL_WANTED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']
LABEL_PRED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']



TRAINING_DATA_FILE = '/scratch/bboy/cchen4/Kepler88/data/kepler88_obs_2016_para2013_Data_3_12_tdv_0.5_average_corrected.pkl' 

torch.manual_seed(0)
np.random.seed(0)

cwd = os.getcwd()
cwd = str(cwd)

start_time = time.time()
main_path = '/scratch/bboy/cchen4/Kepler88/8_phoenix' 
sys.path.insert(1, main_path)

from models.gru_vis_trans import TransAm

######## check device ########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device
print(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

######## read training data ########
training_percent = 0.98 
test_percent = 0.01 

input1_, input2_, target_, label_min, label_max, label_mean,label_std, input_scale_dic = get_train_data(TRAINING_DATA_FILE,
                                                                                                        LABEL_WANTED,
                                                                                                        scale_input=SCALE_INPUT, 
                                                                                                        scale_output=SCALE_OUTPUT, 
                                                                                                        train_percent=training_percent
                                                                                                        )

second_input_size = 0
print('input1 shape:', input1_.shape)
if input2_ is not None:
  second_input_size = input2_.shape[2]
  print('input2 shape:', input2_.shape)
  print(input2_[0])

print('target shape:', target_.shape)

for i in range(len(LABEL_PRED)):
  print(f'max true label of {LABEL_PRED[i]}: {label_max[i]}, min true label of {LABEL_PRED[i]}: {label_min[i]}')
  print(f'max scaled {LABEL_PRED[i]}: {np.max(target_[:,i])}, min scaled {LABEL_PRED[i]}: {np.min(target_[:,i])}')

######## convert data to tensor and ready for training ######## 

num_training_data = training_percent * len(input1_)
if input2_ is None:
  torch_dataset_train, torch_dataset_eva, input1_test_toTensor, target_test = get_tensor_training_data(training_percent, test_percent, input1_, input2_, target_)
else:
  torch_dataset_train, torch_dataset_eva, input1_test_toTensor, input2_test_toTensor, target_test = get_tensor_training_data(training_percent, test_percent, input1_, input2_, target_)

####### Hyper Parameters ########
n_layers_gru =4
n_layers_att =3
feature_size = 256
NHEAD = 4
n_layers_deco =4
BATCH_SIZE =64
EPOCHS = 70
has_gru = True 
LR =0.0008
dropout = 0.1
INIT_LINEAR = 'xav_uni'
INIT_RNN = 'ortho'
WARMUP = True
num_total_step = int(num_training_data / BATCH_SIZE * EPOCHS + 100)
warmup_per = 0.06
warmup_steps = int(warmup_per * (num_total_step))
multi_deco = True
Min_LR = 1e-4


label_pred = LABEL_PRED


######## test model ########
model_test = TransAm(n_layers_gru, 
                    n_layers_att, 
                    feature_size,
                    NHEAD, 
                    dropout=dropout, 
                    num_labels=len(LABEL_PRED), 
                    n_layers_deco=n_layers_deco,
                    multi_deco=multi_deco,
                    init_linear_w=INIT_LINEAR,
                    init_rnn_w=INIT_RNN).to(device)
state_dict = torch.load(cwd+'/model_para_4.pt')
model_test.load_state_dict(state_dict)

# TESTING

preds = predict(model_test, 
                input1_test_toTensor, 
                two_input, 
                label_pred, 
                device, 
                input2_test_toTensor)

pred_val = predict_from_obs(input_scale_dic,
                            model_test,
                            two_input,
                            label_pred,
                            device,
                            SCALE_INPUT,
                            SCALE_OUTPUT,
                            label_max,
                            label_min)

pred_val_inter = predict_from_obs_inter(input_scale_dic,
                            model_test,
                            two_input,
                            label_pred,
                            device,
                            SCALE_INPUT,
                            SCALE_OUTPUT,
                            label_max,
                            label_min)

pred_val_GRIT = predict_from_obs_GRIT(input_scale_dic,
                            model_test,
                            two_input,
                            label_pred,
                            device,
                            SCALE_INPUT,
                            SCALE_OUTPUT,
                            label_max,
                            label_min)

frac_err_list = []
abs_err_list = []

mean_frac_err_list = []
med_frac_err_list = []
std_frac_err_list = []

mean_abs_err_list = []
med_abs_err_list = []
std_abs_err_list = []

if SCALE_OUTPUT == 'MinMax':
    true_label = (target_test + 1) / 2 * (label_max - label_min) + label_min
    true_preds = (preds + 1) / 2 * (label_max - label_min) + label_min

elif SCALE_OUTPUT == 'MinMax01':
    true_label =  target_test * (label_max - label_min) + label_min
    true_preds = preds * (label_max - label_min) + label_min
else:
    true_label = target_test * label_max
    true_preds = preds * label_max


true_label[:, 1] /= Earth_in_Solar
true_preds[:, 1] /= Earth_in_Solar


x_labels = ['a (AU)', 'Mass ($M_E$)', 'e', 'i (deg.)']
name_labels =['a', 'Mass', 'e', 'i']

fig, axs1 = plt.subplots(2, 2,figsize=(14, 10))
print(len(x_labels), preds.shape)

num_bins = 100
bins_list = []


for i in range(len(true_label[0,:])):
    bins_list.append(np.linspace(true_label[:,i].min(), true_label[:, i].max(), num_bins + 1))



for i in range(len(preds[0])):
  ax1 = axs1[i // 2][i % 2]
  frac_err_temp = np.absolute((true_label[:, i] - true_preds[:, i])/true_label[:,i])
  abs_err_temp = np.absolute((true_label[:, i] - true_preds[:, i]))

  if PLOT_FIG == "mad":
    x = true_label[:, i]

    bins = bins_list[i]
    y = frac_err_temp
    medians = []
    mads = []
    bin_centers = []

    temp_values_list = []
    for j in range(num_bins):
      bin_mask = (x >= bins[j]) & (x < bins[j+1])
      bin_x = x[bin_mask]
      bin_y = y[bin_mask]
      
      if len(bin_y) > 0:
          median_y = np.median(bin_y)
          mad_y = np.median(np.abs(bin_y - median_y))
          
          medians.append(median_y)
          mads.append(mad_y)
          bin_centers.append((bins[j] + bins[j + 1]) / 2)
    
    medians = np.array(medians)
    mads = np.array(mads)
    bin_centers = np.array(bin_centers)
      
    if i == 3:
      bin_centers = bin_centers * 180 /np.pi
    ax1.plot(bin_centers, medians)
    ax1.fill_between(bin_centers, medians - mads, medians + mads, alpha=0.2)
    ax1.set_xlabel('True ' + name_labels[i] + ' ' + x_labels[i], fontsize=15)
    ax1.set_ylabel('Median Absolute error')
    ax1.set_yscale('log')


  elif PLOT_FIG == "hist":
    ax1.hist(np.log10(frac_err_temp), bins=50, log=True, density=True)
    
    ax1.set_xlabel('log(' + name_labels[i] + ' fractional error)', fontsize=15)
    
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
  
  elif PLOT_FIG == "scatter":
      if i == 3:
         true_label[:, i] *= 180/np.pi 
      ax1.plot(true_label[:, i], frac_err_temp, 'o', markersize=2, alpha=0.5)
      ax1.set_yscale('log')
      # ax1.set_title('Predicted ' + label_pred[i], fontsize=15)
      ax1.set_xlabel('True '+ x_labels[i], fontsize=15)
      ax1.set_ylabel('Fractional Difference', fontsize=15)
      ax1.tick_params(axis='x', labelsize=14)
      ax1.tick_params(axis='y', labelsize=14)
     

  mean_frac_err_temp = round(np.mean(frac_err_temp), 6)
  med_frac_err_temp = round(np.median(frac_err_temp), 6)
  std_frac_err_temp = round(np.std(frac_err_temp), 6)

  mean_abs_err_temp = round(np.mean(abs_err_temp), 6)
  med_abs_err_temp = round(np.median(abs_err_temp), 6)
  std_abs_err_temp = round(np.std(abs_err_temp), 6)

  frac_err_list.append(frac_err_temp)
  abs_err_list.append(abs_err_temp)

  mean_frac_err_list.append(mean_frac_err_temp)
  med_frac_err_list.append(med_frac_err_temp)
  std_frac_err_list.append(std_frac_err_temp)

  mean_abs_err_list.append(mean_abs_err_temp)
  med_abs_err_list.append(med_abs_err_temp)
  std_abs_err_list.append(std_abs_err_temp)

if PLOT_FIG =="mad":
  plt.savefig(cwd + '/frac_err_mad_Pratio.png', bbox_inches='tight', dpi=300)
elif PLOT_FIG =="hist":
  plt.savefig(cwd + '/frac_err_hist.png', bbox_inches='tight', dpi=300)
elif PLOT_FIG == "scatter":
  plt.savefig(cwd + '/frac_err_scatter.png', bbox_inches='tight', dpi=300)
plt.close()

total_mean_frac_err = sum(mean_frac_err_list)
print('mean frac err:', mean_frac_err_list)#, sec_mae, third_mae, four_mae, five_mae)
print('med frac err:',  med_frac_err_list)#, sec_mae_med, third_mae_med, four_mae_med, five_mae_med)
print(f'std of frac err: {std_frac_err_list}')
print('mean abs err:', mean_abs_err_list)
print('med abs err:', med_abs_err_list)
print(f'std of abs err: {std_abs_err_list}')

print('total time used:', (time.time() - start_time)/60/60) # in hours

