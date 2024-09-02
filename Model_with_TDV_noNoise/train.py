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
from train_utils import * 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys 
from torchsummary import summary
import time
from sklearn.preprocessing import MinMaxScaler
import transformers as tf
from timeit import default_timer as timer
from models.gru_vis_trans import TransAm

cwd = os.getcwd()
cwd = str(cwd)

last_epoch = -1
Continue_train = True # # If want to resume previous training progress, set True, this setting is due to limited runtime/resource
pre_epoch = -1
if Continue_train:
   train_dict = torch.load(cwd+'/checkpoint_3.pth')
   pre_epoch = train_dict['epoch']
   last_epoch = train_dict['last_epoch']
   last_optimizer = train_dict['optimizer_state_dict']
   last_lr = train_dict['lr']
#### training data ######
INCLUDE_APM = False
two_input = True
Earth_in_Solar = 3.0027e-6 # mass
Jupiter_in_Solar = 9.547919e-4 # mass

SCALE_INPUT = 'default'
SCALE_OUTPUT = 'MinMax01'

LABEL_WANTED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']
LABEL_PRED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']

"""
ADD YOUR DATA PATH BELOW AT TRAINING_DATA_FILE
"""
TRAINING_DATA_FILE = '[[YOUR PATH]]/kepler88_obs_2016_para2013_Data_3_12_tdv_0.5_average_corrected.pkl' 

torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()

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
EPOCHS = 200
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

case_name = 'BATCH_' + str(BATCH_SIZE) + '_LR_' + str(LR) + \
            '_nhead_' + str(NHEAD) + '_fSize_' + str(feature_size) + \
            '_nDeco_' + str(n_layers_deco) + '_nEnco_' + str(n_layers_att) + \
            '_dropout_' + str(dropout) + '_nGRU_' + str(n_layers_gru) + \
            '_inScale_' + str(SCALE_INPUT) + '_outScale_' + str(SCALE_OUTPUT) + \
            '_initLinear_' + INIT_LINEAR + '_initRNN_' + INIT_RNN + '_warmup_' + str(WARMUP) + \
            '_multiDeco_' + str(multi_deco) + '_warmupPerc_' + str(warmup_per) + '_minLR_' + str(Min_LR)
writer = SummaryWriter(cwd + '/runs_200/' + case_name)

train_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size= BATCH_SIZE, shuffle=True)
eva_loader = Data.DataLoader(dataset=torch_dataset_eva, batch_size= BATCH_SIZE, shuffle=True)

######## define model, criterion, etc for training ########

model = TransAm(n_layers_gru, 
                n_layers_att, 
                feature_size,
                NHEAD, 
                dropout=dropout, 
                num_labels=len(LABEL_PRED), 
                n_layers_deco=n_layers_deco,
                multi_deco=multi_deco,
                init_linear_w=INIT_LINEAR,
                init_rnn_w=INIT_RNN).to(device)
if Continue_train:
  state_dict = torch.load(cwd+'/model_para_3.pt')
  model.load_state_dict(state_dict)
######## print model parameters #########
print_model_para(model)
######## print model parameters #########


criterion = nn.MSELoss() #loss function, mean square error


if Continue_train:
   optimizer = torch.optim.AdamW(model.parameters(), lr=last_lr)
   optimizer.load_state_dict(last_optimizer)
else:
   optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

if WARMUP:
  scheduler = tf.get_linear_schedule_with_warmup(optimizer, warmup_steps, num_total_step, last_epoch=last_epoch)
  # scheduler2 = StepLR(optimizer, step_size=200, gamma=0.99)
else:
  scheduler = StepLR(optimizer, step_size=200, gamma=0.99, last_epoch=last_epoch)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =============================#
####    train the model     ####
# =============================#
label_pred = LABEL_PRED

last_loss, eval_loss = train_and_eval(train_loader, 
                            eva_loader, 
                            model, 
                            criterion, 
                            optimizer, 
                            scheduler, 
                            Min_LR,
                            writer, 
                            EPOCHS,
                            LR,
                            device,
                            label_pred, 
                            two_input,
                            input_scale_dic,
                            SCALE_INPUT,
                            SCALE_OUTPUT,
                            label_max,
                            label_min,
                            pre_epoch)

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
state_dict = torch.load(cwd+'/model_para.pt')
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

x_labels = ['AU', 'M_Earth', '', 'rad']


fig, axs1 = plt.subplots(7, 2,figsize=(25, 55))
print(len(x_labels), preds.shape)
for i in range(len(preds[0])):
  
  frac_err_temp = np.absolute((true_label[:, i] - true_preds[:, i])/true_label[:,i])
  abs_err_temp = np.absolute((true_label[:, i] - true_preds[:, i]))

  
  ax1 = axs1[i // 2][i % 2]
  ax1.plot(true_label[:, i], frac_err_temp, 'o', markersize=2, alpha=0.5)
  ax1.set_yscale('log')
  ax1.set_title('Predicted ' + label_pred[i], fontsize=15)
  ax1.set_xlabel(x_labels[i], fontsize=15)
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

plt.savefig(cwd + '/frac_err.png', bbox_inches='tight')
plt.close()

fig, axs1 = plt.subplots(7, 2,figsize=(25, 55))
for i in range(len(preds[0])):

  ax1 = axs1[i // 2][i % 2]
  ax1.plot(true_label[:, i], abs_err_list[i], 'o', markersize=2, alpha=0.5)
  ax1.set_yscale('log')
  ax1.set_title('Predicted ' + label_pred[i], fontsize=15)
  ax1.set_xlabel('True Label ' + x_labels[i], fontsize=15)
  ax1.set_ylabel('Abs Difference', fontsize=15)
  ax1.tick_params(axis='x', labelsize=14)
  ax1.tick_params(axis='y', labelsize=14)

plt.savefig(cwd + '/abs_err.png', bbox_inches='tight')
plt.close()

total_mean_frac_err = sum(mean_frac_err_list)
print('mean frac err:', mean_frac_err_list)
print('med frac err:',  med_frac_err_list)
print(f'std of frac err: {std_frac_err_list}')
print('mean abs err:', mean_abs_err_list)
print('med abs err:', med_abs_err_list)
print(f'std of abs err: {std_abs_err_list}')



new_file = open(str(last_loss)+'_'+ str(eval_loss) + '_' +str(total_mean_frac_err) +'.txt', 'w')
new_file.close()

print('total time used:', (time.time() - start_time)/60/60) # in hours

