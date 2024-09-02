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

def get_nontransit_a(
    m_t, # mass of transit planet
    m_nont, # mass of non-transit planet 
    m_star, # mass of star
    a_t, # semi-major axis of transit planet
    n_Rhill, # least number of Rhill needed from the transit planet
    locate_in_or_out, # non-transit is inside/ouside of the transit one or both: 'in', 'out', 'both' 
    has_range # select a range for the non-transit a or not
    ):
    DaysOfYear = 365.25
    X = ((m_t + m_nont)/(3*m_star))**(1/3)
    Rh_outer = X * a_t/(1 - n_Rhill/2*X)
    Rh_inner = X * a_t/(1 + n_Rhill/2*X)

    a_nontransit_outer_bound = a_t + n_Rhill*Rh_outer 
    a_nontransit_inner_bound = a_t - n_Rhill*Rh_inner 

    if has_range:
        a_10day = (m_star * (10/DaysOfYear)**2)**(1/3)
        a_100day = (m_star * (100/DaysOfYear)**2)**(1/3)

        a_nontransit_outer = np.random.uniform(a_nontransit_outer_bound, a_100day)
        a_nontransit_inner = np.random.uniform(a_10day, a_nontransit_inner_bound) 
        
        if locate_in_or_out == 'both':
            total_a_range = a_100day - a_nontransit_outer_bound + a_nontransit_inner_bound - a_10day
            p_outer = (a_100day - a_nontransit_outer_bound)/total_a_range 
            p_inner = (a_nontransit_inner_bound - a_10day)/total_a_range

            a_nontransit = np.random.choice([a_nontransit_outer, a_nontransit_inner], p=[p_outer, p_inner])

            return a_nontransit
        
        elif locate_in_or_out =='in':
            return a_nontransit_inner 

        elif locate_in_or_out =='out':
            return a_nontransit_outer 

    else:
        if locate_in_or_out == 'both':
            a_nontransit = np.random.choice([a_nontransit_outer_bound, a_nontransit_inner_bound])
            return a_nontransit

        elif locate_in_or_out == 'in':
            return a_nontransit_inner_bound

        elif locate_in_or_out == 'out':
            return a_nontransit_outer_bound
#### training data ######
INCLUDE_APM = False
two_input = True
Earth_in_Solar = 3.0027e-6 # mass
Jupiter_in_Solar = 9.547919e-4 # mass

SCALE_INPUT = 'default'
SCALE_OUTPUT = 'MinMax01'

LABEL_WANTED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']
LABEL_PRED = ['a_nont', 'm_nont', 'e_nont', 'incli_nont']
              # 'm_t', 'm_nont', 
              # 'e_nont',
              # 'i_t', 'i_nont', 
              # 'cos(o_t)', 'sin(o_t)', 'cos(o_nont)', 'sin(o_nont)', 
              # 'cos(O_nont)', 'sin(O_nont)', 
              # 'cos(Mean_ano_nont)', 'sin(Mean_ano_nont)']

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

if SCALE_OUTPUT == 'MinMax':
    true_label = (target_test + 1) / 2 * (label_max - label_min) + label_min

elif SCALE_OUTPUT == 'MinMax01':
    true_label =  target_test * (label_max - label_min) + label_min
else:
    true_label = target_test * label_max


print(target_test.shape)
criterion_1 = 0.01
condition_1 = (abs(true_label[:, 0] - 3) < 3*criterion_1) & (abs(true_label[:, 1] - Jupiter_in_Solar) < Jupiter_in_Solar *criterion_1)
indices_1 = np.array(np.nonzero(condition_1))[0]
print(indices_1.shape)

criterion_2 = 0.0039
condition_2 = (abs(true_label[:, 0] - 3) < 3*criterion_2) & (abs(true_label[:, 1] - 5 * Jupiter_in_Solar) < 5 * Jupiter_in_Solar *criterion_2)
indices_2 = np.array(np.nonzero(condition_2))[0]
indices_2 = indices_2[0]
indices_2 = indices_2.reshape(1,)
print(indices_2.shape)


criterion_3 = 0.15
condition_3 = (abs(true_label[:, 0] - 0.2) < 0.2 * criterion_3) & (abs(true_label[:, 1] - 17 * Earth_in_Solar) < 17 * Earth_in_Solar *criterion_3)
indices_3 = np.array(np.nonzero(condition_3))[0]
print(indices_3.shape)
# exit()

input1_test_1 = input1_test_toTensor[indices_1]
input2_test_1 = input2_test_toTensor[indices_1]
true_label_1 = true_label[indices_1]

input1_test_2 = input1_test_toTensor[indices_2]
input2_test_2 = input2_test_toTensor[indices_2]
true_label_2 = true_label[indices_2]

input1_test_3 = input1_test_toTensor[indices_3]
input2_test_3 = input2_test_toTensor[indices_3]
true_label_3 = true_label[indices_3]
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

label_pred = LABEL_PRED

preds_1 = predict(model_test, 
                  input1_test_1, 
                  two_input, 
                  label_pred, 
                  device, 
                  input2_test_1)

preds_2 = predict(model_test, 
                  input1_test_2, 
                  two_input, 
                  label_pred, 
                  device, 
                  input2_test_2)

preds_3 = predict(model_test, 
                  input1_test_3, 
                  two_input, 
                  label_pred, 
                  device, 
                  input2_test_3)

if SCALE_OUTPUT == 'MinMax':
    true_preds_1 = (preds_1 + 1) / 2 * (label_max - label_min) + label_min
    true_preds_2 = (preds_2 + 1) / 2 * (label_max - label_min) + label_min
    true_preds_3 = (preds_3 + 1) / 2 * (label_max - label_min) + label_min

elif SCALE_OUTPUT == 'MinMax01':
    true_preds_1 = preds_1 * (label_max - label_min) + label_min
    true_preds_2 = preds_2 * (label_max - label_min) + label_min
    true_preds_3 = preds_3 * (label_max - label_min) + label_min
else:
    true_preds_1 = preds_1 * label_max
    true_preds_2 = preds_2 * label_max
    true_preds_3 = preds_3 * label_max


frac_err_1 = abs(true_label_1 - true_preds_1)/true_label_1
frac_err_2 = abs(true_label_2 - true_preds_2)/true_label_2
frac_err_3 = abs(true_label_3 - true_preds_3)/true_label_3

scale_12 = np.array([1, Jupiter_in_Solar, 1, np.pi/180])
scale_3 = np.array([1, Earth_in_Solar, 1, np.pi/180])

print(f'For 3AU ~ 1 M_J case, frac err is {frac_err_1}. \n \
      The true values are {true_label_1/scale_12} \n \
      The predicted value are {true_preds_1/scale_12}.')
    #   The predicted a is {true_preds_1[0, 0]} AU, and true val is {true_label_1[0, 0]}. \n \
    #   The predicted mass is {true_preds_1[0, 1]/Jupiter_in_Solar} M_J, and true val is {true_label_1[0, 1]/Jupiter_in_Solar} M_J.\n \
    #   The predicted e is {true_preds_1[0, 2]} AU, and true val is {true_label_1[0, 2]}.\n \
    #   The predicted inclination is {true_preds_1[0, 3]*180/np.pi} deg, and true val is {true_label_1[0, 3]*180/np.pi} deg.')

print(f'For 3AU ~ 5 M_J case, frac err is {frac_err_2}.\n \
      The true values are {true_label_2/scale_12} \n \
      The predicted value are {true_preds_2/scale_12}.')
    #   The predicted a is {true_preds_2[0, 0]} AU, and true val is {true_label_2[0, 0]}.\n \
    #   The predicted mass is {true_preds_2[0, 1]/Jupiter_in_Solar} M_J, and true val is {true_label_2[0, 1]/Jupiter_in_Solar} M_J.\n \
    #   The predicted e is {true_preds_2[0, 2]} AU, and true val is {true_label_2[0, 2]}.\n \
    #   The predicted inclination is {true_preds_2[0, 3]*180/np.pi} deg, and true val is {true_label_2[0, 3]*180/np.pi} deg.')

print(f'For 0.3 AU ~ 15 M_E case, frac err is {frac_err_3}.\n \
      The true values are {true_label_3/scale_3} \n \
      The predicted value are {true_preds_3/scale_3}.')
    #   The predicted a is {true_preds_3[:, 0]} AU, and true val is {true_label_3[:, 0]}.\n \
    #   The predicted mass is {true_preds_3[:, 1]/Earth_in_Solar} M_E, and true val is {true_label_3[:, 1]/Earth_in_Solar} M_E.\n \
    #   The predicted e is {true_preds_3[:, 2]} AU, and true val is {true_label_3[:, 2]}.\n \
    #   The predicted inclination is {true_preds_3[:, 3]*180/np.pi} deg, and true val is {true_label_3[:, 3]*180/np.pi} deg.')