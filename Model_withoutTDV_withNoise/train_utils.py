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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys 
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device


labels_all = {'m_t':0, 'm_nont':1, 
            'e_t': 2, 'e_nont': 3, 
            'incli_t': 4, 'incli_nont': 5, 
            'omega_t': 6, 'omega_nont': 7, 
            'Omega_t': 8, 'Omega_nont': 9, 
            'Mean_ano_t': 10, 
            'Mean_ano_nont': 11,
            'a_nont': 12}


def print_model_para(model):
#   table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
    #   table.add_row([name, params])
      total_params+=params
#   print(table)
  print(f"Total Trainable Params: {total_params}")

def get_train_data(data_path, 
                   label_wanted,
                   scale_input, 
                   scale_output, 
                   train_percent,
                   NO_TDV
                   #include_apm=False# a->semi-major axis, p->period, m->star mass
                   ): 

    with open(data_path, 'rb') as fileo:
        datao = pickle.load(fileo)
    fileo.close()

    print('Finished reading data from pkl')

    # num of data used
    # num_data = 6500
    ttv = datao[0]*24 # day to hr
    tdv = datao[1]  # percentage
    duration = datao[2]
    transit_time = datao[3]/365.25 # convert transit time from day to yr
    fitted_P = datao[4]
    labels_data = datao[-1]

    average_dur = np.mean(duration, axis=1)

    # print(f'ttv min is {np.min(ttv)*60}, max is {np.max(ttv)*60}')
    # print(f'tdv min is {np.min(tdv)}, max is {np.max(tdv)}')
    # print(f'dur min is {np.min(duration)}, max is {np.max(duration)}')
    # print(f'transit time min is {np.min(transit_time)*365.25}, max is {np.max(transit_time)*365.25}')
    # print(f'fitted P min is {np.min(fitted_P)}, max is {np.max(fitted_P)}')
    # exit()
    ############

    training_num = int(len(ttv) * train_percent)

    labels = None 

    for label_name in label_wanted:
        label_idx = labels_all[label_name]
        if label_name == 'm_t' or label_name == 'm_nont' or label_name == 'e_t' or label_name == 'e_nont' or label_name =='a_nont':
            labels_temp = labels_data[:, label_idx].reshape(-1, 1)

        elif label_name == 'incli_t' or label_name == 'incli_nont':
            labels_temp = labels_data[:, label_idx].reshape(-1, 1) #np.cos(labels_data[:, label_idx].reshape(-1, 1))
        else:
            labels_temp_cos = np.cos(labels_data[:, label_idx].reshape(-1, 1))
            labels_temp_sin = np.sin(labels_data[:, label_idx].reshape(-1, 1))
            labels_temp = np.append(labels_temp_cos, labels_temp_sin, axis=1)

        if labels is None:
            labels = labels_temp 
            
        else:
            labels = np.append(labels, labels_temp, axis=1)

    ttv_min = np.amin(ttv[:training_num, :])
    ttv_max = np.amax(ttv[:training_num, :])
    # ttv_mean = np.mean(ttv, axis=0) # shape: (20,)
    # ttv_std = np.std(ttv, axis=0)

    tdv_min = np.min(tdv[:training_num, :])
    tdv_max = np.max(tdv[:training_num, :])  
    # tdv_mean = np.mean(tdv, axis=0)
    # tdv_std = np.std(tdv, axis=0)
    average_dur_max = np.max(average_dur[:training_num])
    average_dur_min = np.min(average_dur[:training_num]) 

    fitted_P_max = np.max(fitted_P[:training_num])
    fitted_P_min = np.min(fitted_P[:training_num])
    
    average_dur = 2 * (average_dur - average_dur_min)/(average_dur_max - average_dur_min) - 1
    fitted_P = 2 * (fitted_P - fitted_P_min)/(fitted_P_max - fitted_P_min) - 1

    input_scale_dic = {}
    input_scale_dic['ttv_min'] = ttv_min 
    input_scale_dic['ttv_max'] = ttv_max 
    # input_scale_dic['ttv_mean'] = ttv_mean 
    # input_scale_dic['ttv_std'] = ttv_std

    input_scale_dic['tdv_min'] = tdv_min 
    input_scale_dic['tdv_max'] = tdv_max 
    # input_scale_dic['tdv_mean'] = tdv_mean 
    # input_scale_dic['tdv_std'] = tdv_std  
    
    input_scale_dic['average_dur_max'] = average_dur_max
    input_scale_dic['average_dur_min'] = average_dur_min

    input_scale_dic['average_P_min'] = fitted_P_min
    input_scale_dic['average_P_max'] = fitted_P_max

    transit_time_min = np.min(transit_time[:training_num, :])
    transit_time_max = np.max(transit_time[:training_num, :])

    input_scale_dic['transit_time_min'] = transit_time_min
    input_scale_dic['transit_time_max'] = transit_time_max


    labels_min = np.amin(labels[:training_num, :], axis=0)
    labels_max = np.amax(labels[:training_num, :], axis=0)
    labels_mean = np.mean(labels[:training_num, :], axis=0)
    labels_std = np.std(labels[:training_num, :], axis=0)

    print(f'labels_min: {list(labels_min)}')
    print(f'labels_max: {list(labels_max)}')
    print(f'ttv_min: {ttv_min}, ttv_max: {ttv_max}')
    print(f'tdv_min: {tdv_min}, tdv_max: {tdv_max}')
    print(f'transit_time_min={transit_time_min}, transit_time_max={transit_time_max}')

    if scale_input == 'MinMax':
        ttv = 2 * (ttv - ttv_min)/(ttv_max - ttv_min) - 1
        tdv = 2 * (tdv - tdv_min)/(tdv_max - tdv_min) - 1
        transit_time = 2 * (transit_time - transit_time_min)/(transit_time_max - transit_time_min) - 1

    elif scale_input == 'MinMax01':
        ttv = (ttv - ttv_min)/(ttv_max - ttv_min)
        tdv = (tdv - tdv_min)/(tdv_max - tdv_min) 
        transit_time = (transit_time - transit_time_min)/(transit_time_max - transit_time_min)     
    else: #default
        tdv *= 100


    
    if scale_output == 'MinMax01':
        labels = (labels - labels_min)/(labels_max - labels_min)
    elif scale_output == 'MinMax':
        labels = 2 * (labels - labels_min)/(labels_max - labels_min) - 1
    else:
        labels = labels/labels_max




    print('After scaling max ttv: ', np.max(ttv), ' min ttv: ', np.min(ttv))
    print('max transit time: ', np.max(transit_time), ' min transit_time: ', np.min(transit_time))
    print('max scaled tdv: ', np.max(tdv), ' min scaled tdv: ', np.min(tdv))

    if NO_TDV:
        input1 = np.stack([ttv, transit_time], axis=2)
    else:
        input1 = np.stack([ttv, tdv, transit_time], axis=2)


    average_dur = average_dur.reshape(-1, 1)
    fitted_P = fitted_P.reshape(-1, 1)

    input2 = np.stack([average_dur, fitted_P], axis=2)

    return input1, input2, labels, labels_min, labels_max, labels_mean, labels_std, input_scale_dic

def get_tensor_training_data(training_percent, 
                            test_percent,
                            input1_, 
                            input2_, 
                            target_
                            ):
    training_num = int(len(input1_) * training_percent)
    test_num = int(len(input1_) * test_percent)

    # split test and trainning set
    input1_train = input1_[:training_num]
    target_train = target_[:training_num]

    input1_eval = input1_[training_num:-test_num]
    target_eval = target_[training_num:-test_num]

    input1_test = input1_[-test_num:]
    target_test = target_[-test_num:]

    input_train_toTensor1 = torch.from_numpy(input1_train).float()
    target_train_toTensor = torch.from_numpy(target_train).float()

    input_eva_toTensor1 = torch.from_numpy(input1_eval).float()
    target_eva_toTensor = torch.from_numpy(target_eval).float()

    input_test_toTensor1 = torch.from_numpy(input1_test).float()

    print(f'data shape is {input_train_toTensor1.shape}, {input_eva_toTensor1.shape}, {input_test_toTensor1.shape}')
    if input2_ is None:
        torch_dataset_train = Data.TensorDataset(input_train_toTensor1, target_train_toTensor)
        torch_dataset_eva = Data.TensorDataset(input_eva_toTensor1,  target_eva_toTensor)

    
        return torch_dataset_train, torch_dataset_eva, input_test_toTensor1, target_test
    else: 
        input2_train = input2_[:training_num]
        input2_eval = input2_[training_num:-test_num]
        input2_test = input2_[-test_num:]

        input_eva_toTensor2 = torch.from_numpy(input2_eval).float()

        input_train_toTensor2 = torch.from_numpy(input2_train).float()

        input_test_toTensor2 = torch.from_numpy(input2_test).float()

        torch_dataset_train = Data.TensorDataset(input_train_toTensor1, 
                                                input_train_toTensor2,
                                                target_train_toTensor)
        torch_dataset_eva = Data.TensorDataset(input_eva_toTensor1, 
                                            input_eva_toTensor2,
                                            target_eva_toTensor)
        return (torch_dataset_train, 
               torch_dataset_eva, 
               input_test_toTensor1, 
               input_test_toTensor2,
               target_test)


def validation(eva_loader, model, criterion, device, two_input):
    total_loss_eva = 0

    with torch.no_grad():
# one input
        if not two_input:
            for eva_step, (x1_eva, y_eva) in enumerate(eva_loader):

                output_eva = model(x1_eva.to(device))

                loss_eva = criterion(output_eva, y_eva.to(device))
                total_loss_eva += loss_eva
        else:
            for eva_step, (x1_eva, x2_eva, y_eva) in enumerate(eva_loader):

                output_eva = model(x1_eva.to(device), x2_eva.to(device))

                loss_eva = criterion(output_eva, y_eva.to(device))
                total_loss_eva += loss_eva            

    total_loss_eva = total_loss_eva / (1 + eva_step)

    return total_loss_eva, output_eva, y_eva

def train_and_eval(train_loader,
                   eva_loader, 
                   model, 
                   criterion, 
                   optimizer, 
                   scheduler, 
                   min_lr,
                   writer, 
                   EPOCH,
                   LR,
                   device,
                   label_pred,
                   two_input,
                   input_scale_dic,
                   SCALE_INPUT,
                   SCALE_OUTPUT,
                   label_max,
                   label_min,
                   NO_TDV):

    eval_loss_list = []

    count_step = 0
    loss = None
    cwd = os.getcwd()
    cwd = str(cwd)
    lr_curve_file = open(cwd + '/lr_curve.txt', 'w')
    eva_error_file = open(cwd + '/eva_eror.txt', 'w')

# currently version only one input
    for epoch in range(EPOCH):
        for step, (x1, x2, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            noise_ttv = np.random.normal(loc=0.0, scale=1.0/60., size=x1[:, :, 0].shape)
            noise_transit = np.random.normal(loc=0.0, scale=1.0/60./24/365.25, size=x1[:, :, 1].shape)
            noise = np.stack([noise_ttv, noise_transit], axis=2)
            # add noise here
            x1 += torch.from_numpy(noise).float() 
            
            output = model(x1.to(device), x2.to(device))
        
            loss = criterion(output, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            # # fetch current lr
            # for param_group in optimizer.param_groups:
            #     cur_lr = param_group['lr'] 

            # if epoch < 10 or (epoch >= 10 and cur_lr > min_lr):
            scheduler.step()

            count_step += 1

            if step % 100 == 0:
                print('| epoch {:3d} | step {:3d}| loss {:5.5f}'.format(epoch, step, loss))
                print('lr:')
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                ######## write to tensorboard #######
                writer.add_scalar('train loss', loss.item(), count_step)
                model.eval()
                total_loss_eva, output_eva, y_eva = validation(eva_loader, model, criterion, device, two_input)
                ######## write to tensorboard ########
                writer.add_scalar('Eva loss', total_loss_eva, count_step)
        
        ######## end of each epoch, do evaluation ########
        model.eval()
        total_loss_eva, output_eva, y_eva = validation(eva_loader, model, criterion, device, two_input)
        eval_loss_list.append(total_loss_eva)
        output_eva_np = np.array(output_eva.cpu())
        label_eva_np = np.array(y_eva)
        pred_val_grit = predict_from_obs_GRIT(input_scale_dic,
                                    model,
                                    two_input,
                                    label_pred,
                                    device,
                                    SCALE_INPUT,
                                    SCALE_OUTPUT,
                                    label_max,
                                    label_min,
                                    NO_TDV)
        pred_val_inter = predict_from_obs_inter(input_scale_dic,
                                                model,
                                                two_input,
                                                label_pred,
                                                device,
                                                SCALE_INPUT,
                                                SCALE_OUTPUT,
                                                label_max,
                                                label_min,
                                                NO_TDV)
        pred_val_NOinter = predict_from_obs(input_scale_dic,
                                        model,
                                        two_input,
                                        label_pred,
                                        device,
                                        SCALE_INPUT,
                                        SCALE_OUTPUT,
                                        label_max,
                                        label_min,
                                        NO_TDV)
        
        for kk in range(len(label_pred)):
            temp_error = np.mean(np.absolute((output_eva_np[:, kk] - label_eva_np[:, kk])/label_eva_np[:, kk]))
            writer.add_scalar(label_pred[kk] + ' eva error', temp_error, count_step)
            
            writer.add_scalar(f'pred obs from GRIT {label_pred[kk]}', pred_val_grit[0][kk], count_step)
            writer.add_scalar(f'pred obs inter {label_pred[kk]}', pred_val_inter[0][kk], count_step)
            writer.add_scalar(f'pred obs nointer {label_pred[kk]}', pred_val_NOinter[0][kk], count_step)

        writer.flush()

        eva_error_file.write(str(epoch) + ' ' + str(total_loss_eva)+'\n')
        eva_error_file.flush()

        ######## print loss ########
        print('-' * 89)
        print('end of epoch ', epoch, 'loss:', loss, 'eval loss:', total_loss_eva)
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('-' * 89)

        lr_curve_file.write(str(epoch) +' '+str(loss.item()) +'\n')
        lr_curve_file.flush()
        torch.save(model.state_dict(), cwd+'/model_para.pt')

    
    lr_curve_file.close()
    eva_error_file.close()

    last_loss = loss.item()

    return last_loss, total_loss_eva.item()

def predict(model_test, 
            input1_test_toTensor, 
            INCLUDE_APM, 
            label_pred,
             device, 
             input2_test_toTensor=None):

    model_test.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(input1_test_toTensor)):
            val1 = input1_test_toTensor[i]
            val1 = val1.reshape(1, val1.shape[0], val1.shape[1])
            if INCLUDE_APM:
                val2 = input2_test_toTensor[i]
                val2 = val2.reshape(1, val2.shape[0], val2.shape[1])
                y_hat = model_test(val1.to(device), val2.to(device))
            else:
                y_hat = model_test(val1.to(device))

            y_hat = y_hat.reshape(len(label_pred))
            preds.append(y_hat.tolist())


    preds = np.array(preds)

    return preds


def predict_from_obs(input_scale_dic,
                     model,
                     INCLUDE_APM,
                     label_pred,
                     device,
                     SCALE_INPUT,
                     SCALE_OUTPUT,
                     label_max,
                     label_min,
                     NO_TDV
                    ):
    ################ Predcit for the observation ##################
    # mins to hr
    ttv_obs = np.array([583.3, 543.6, 504.2, 456.6, 365.8, 311.8, 260.4, 204.1, 152.2, 
                        35.5, -25.5, -88.1, -149.1, -212.2, -273.6, -338.3, -396.6, -460.7,
                        -512.5, -610.4, -655.0, -709.9, -713.7, -720.4, -703.5, -687.1, -646.6,
                        -602.7, -544.5, -482.2, -404.8, -315.3, -232.6, -126.1, -46.2, 73.9, 153.0, 
                        267.2, 337.3, 441.4, 498.1, 583.4, 622.7, 691.4, 705.2, 745.6, 739.9, 754.9, 
                        734.8, 723.5, 686.8, 582.6, 528.6, 480.4, 423.3, 366.8, 317.2, 261.3, 208.0, 
                        154.6, 103.1, 46.8, -4.5, -55.0, -107.6, -151.4, -196.9, -246.3, -288.4, -333.6, 
                        -377.4, -412.4, -456.2, -485.1, -528.9, -552.8, -571.7, -579.0, -592.9, -592.7, 
                        -565.4, -557.0, -517.4, -484.3, -375.0, -302.2, -241.9, -82.6, -1.4, 84.3, 159.5, 
                        247.1, 314.1, 395.8, 445.2, 515.2, 551.9, 600.6, 624.9, 628.7, 598.3, 584.8, 532.2, 
                        500.0, 437.8, 384.7, 320.2, 198.8, 69.8, 6.4, -46.9, -158.9, -212.2, -255.6, -305.8, 
                        -341.4, -381.6, -416.6, -446.6, -470.8, -495.3])/60
    # percent
    tdv_obs = np.array([-0.058, 0.005, -0.018, -0.009, 0.013, -0.031, -0.01, -0.005, -0.005, -0.043, -0.043, 
                        -0.05, -0.026, -0.047, 0.002, -0.076, -0.043, -0.017, -0.057, -0.041, 0.009, 0.008, 
                        0.004, -0.04, -0.028, -0.036, -0.068, -0.052, -0.052, -0.017, -0.024, 0.052, 0.076, 
                        -0.055, -0.128, 0.027, 0.017, 0.053, -0.045, -0.02, -0.018, 0.027, 0.007, 0.008, 
                        0.082, 0.054, -0.03, 0.023, -0.027, 0.019, 0.039, 0.012, 0.035, 0.017, 0.057, 0.0, 
                        0.035, -0.004, -0.023, 0.026, 0.063, 0.013, 0.01, -0.017, -0.037, 0.002, -0.067,
                        -0.007, 0.004, 0.004, 0.051, 0.026, 0.045, -0.047, -0.067, -0.022, 0.001, -0.023, 
                        -0.014, -0.044, -0.01, 0.003, -0.048, 0.016, -0.044, -0.02, -0.015, 0.021, 0.044, 
                        -0.025, 0.012, 0.007, 0.022, -0.016, 0.029, 0.028, -0.022, 0.016, 0.026, 0.05, 0.032, 
                        .031, 0.006, 0.045, -0.008, -0.006, 0.025, 0.001, -0.003, 0.018, 0.005, 0.026, -0.018, 
                        .013, -0.026, 0.013, -0.009, 0.007, -0.019, 0.027, 0.023])


    transit_time_obs = np.array([79315.6096, 95048.34192, 110781.37424, 126506.20800000003, 157960.27263999998, 
                                 173678.7064, 189399.73872, 205115.87103999997, 220836.4048, 252264.56944, 267976.0032, 283685.83552, 
                                299397.26784000004, 315106.6016, 330817.63392000005, 346525.36624000006, 362239.5, 377947.83232, 
                                393668.46463999996, 425115.43072, 440843.26304000005, 472333.22912, 488101.86288, 503867.5951999999, 
                                519656.92751999997, 535445.76128, 551258.6936, 567075.0259199999, 582905.65968, 598740.392, 
                                614590.2243199999, 630452.15808, 646307.2904, 662186.2227200001, 678038.55648, 693931.0888,
                                 709782.6211199999, 725669.2548799999, 741511.7872, 757388.31952, 773217.45328, 789075.1856000001, 
                                 804886.9179199998, 820728.0516799999, 836514.2839999999, 852327.1163199999, 868093.85008, 883881.2824, 
                                 899633.6147200001, 915394.74848, 931130.4808, 978343.5791999999, 994062.01152, 1009786.24528, 
                                 1025501.5776000001, 1041217.5099200001, 1056940.34368, 1072656.8760000002, 1088376.00832, 
                                 1104095.04208, 1119815.9744000002, 1135532.10672, 1151253.24048, 1166975.1728, 1182695.0051199999, 
                                 1198423.6388800002, 1214150.5712000001, 1229873.6035200001, 1245603.93728, 1261331.1696, 1277059.80192, 
                                 1292797.2356800002, 1308525.868, 1324269.40032, 1339998.0340800001, 1355746.5664, 1371500.0987200001, 
                                 1387265.23248, 1403023.7648, 1418796.39712, 1450368.5632000002, 1466149.39552, 1481961.42928, 
                                 1497766.9615999998, 1529421.12768, 1545266.36, 1561099.09232, 1592803.2584, 1608656.8907199998, 
                                 1624515.0244800001, 1640362.6568, 1656222.6891200002, 1672062.12288, 1687916.2552, 1703738.0875199998, 
                                 1719580.5212800002, 1735389.6535999998, 1751210.7859200004, 1798552.38432, 1814328.6180800002, 
                                 1830070.6504000002, 1845829.5827200003, 1861549.41648, 1877289.6488, 1892999.88112, 1908719.2148799999, 
                                 1924427.1471999998, 1955850.61328, 1987266.47792, 2002975.5116799998, 2018694.644, 2050127.5100800002,
                                  2065846.6424, 2081575.67472, 2097297.90848, 2113034.7408, 2128766.9731199997, 2144504.40688, 2160246.8392, 
                                  2175995.07152, 2191743.00528]) 
    transit_time_obs = (transit_time_obs + 0)/60/24/365.25 # min to yr
   

    ttv_min = input_scale_dic['ttv_min'] 
    ttv_max = input_scale_dic['ttv_max']
    tdv_min = input_scale_dic['tdv_min']
    tdv_max = input_scale_dic['tdv_max']
    transit_time_min = input_scale_dic['transit_time_min']
    transit_time_max = input_scale_dic['transit_time_max']

    average_dur = np.array([3.7285/24]) # in days
    average_P = np.array([10.9530783]) # in days

    average_dur_min = input_scale_dic['average_dur_min']
    average_dur_max = input_scale_dic['average_dur_max']
    average_P_min = input_scale_dic['average_P_min']
    average_P_max = input_scale_dic['average_P_max']

    average_dur = 2 * (average_dur - average_dur_min)/(average_dur_max - average_dur_min) - 1
    average_P = 2 * (average_P - average_P_min)/(average_P_max - average_P_min) - 1

    if SCALE_INPUT == 'MinMax':

        ttv_obs = 2 * (ttv_obs - ttv_min)/(ttv_max - ttv_min) - 1
        tdv_obs = 2 * (tdv_obs - tdv_min)/(tdv_max - tdv_min) - 1
        transit_time_obs = 2 * (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min) - 1


    elif SCALE_INPUT == 'Minmax01':

        ttv_obs = (ttv_obs - ttv_min)/(ttv_max - ttv_min)
        tdv_obs = (tdv_obs - tdv_min)/(tdv_max - tdv_min)
        transit_time_obs = (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min)
    else:
        tdv_obs *= 100



    ttv_obs = ttv_obs.reshape(1, -1)
    transit_time_obs = transit_time_obs.reshape(1, -1)
    tdv_obs = tdv_obs.reshape(1, -1)


    if NO_TDV:
        input1_obs = np.stack([ttv_obs, transit_time_obs], axis=2)
    else:
        input1_obs = np.stack([ttv_obs, tdv_obs, transit_time_obs], axis=2)

    val1 = torch.from_numpy(input1_obs).float()

    input2_obs = np.stack([average_dur.reshape(-1, 1), average_P.reshape(-1, 1)], axis= 2)
    val2 = torch.from_numpy(input2_obs).float()

    pred_obs = predict(model, 
                        val1, 
                        INCLUDE_APM, 
                        label_pred,
                        device,
                        val2)

    if SCALE_OUTPUT == 'MinMax01':
        pred_obs = pred_obs * (label_max - label_min) + label_min
    elif SCALE_OUTPUT == 'MinMax':
        pred_obs = (pred_obs + 1) / 2 * (label_max - label_min) + label_min
    else:
        pred_obs *= label_max

    print(f'The model prediction for observation is {pred_obs} at the end of current epoch.')
    return pred_obs


def predict_from_obs_inter(input_scale_dic,
                     model,
                     INCLUDE_APM,
                     label_pred,
                     device,
                     SCALE_INPUT,
                     SCALE_OUTPUT,
                     label_max,
                     label_min,
                     NO_TDV
                    ):
    ################ Predcit for the observation 2016 ##################
    # mins to hr
    ttv_obs = np.array([583.3, 543.6, 504.2, 456.6, 411.20000000000005, 365.8, 311.8, 260.4, 204.1, 152.2, 93.85000000000002, 
                        35.5, -25.5, -88.1, -149.1, -212.2, -273.6, -338.3, -396.6, -460.7, -512.5, -561.45, -610.4, -655.0, 
                        -682.45, -709.9, -713.7, -720.4, -703.5, -687.1, -646.6, -602.7, -544.5, -482.2, -404.8, -315.3, -232.6, 
                        -126.1, -46.2, 73.9, 153.0, 267.2, 337.3, 441.4, 498.1, 583.4, 622.7, 691.4, 705.2, 745.6, 739.9, 754.9, 
                        734.8, 723.5, 686.8, 652.0666666666668, 617.3333333333335, 582.6, 528.6, 480.4, 423.3, 366.8, 317.2, 261.3, 
                        208.0, 154.6, 103.1, 46.8, -4.5, -55.0, -107.6, -151.4, -196.9, -246.3, -288.4, -333.6, -377.4, -412.4, -456.2, 
                        -485.1, -528.9, -552.8, -571.7, -579.0, -592.9, -592.7, -579.05, -565.4, -557.0, -517.4, -484.3, -429.64999999999964, 
                        -375.0, -302.2, -241.9, -162.25, -82.6, -1.4, 84.3, 159.5, 247.1, 314.1, 395.8, 445.2, 515.2, 551.9, 600.6, 608.7, 
                        616.8000000000001, 624.9, 628.7, 598.3, 584.8, 532.2, 500.0, 437.8, 384.7, 320.2, 259.5, 198.8, 134.30000000000018, 
                        69.8, 6.4, -46.9, -102.89999999999964, -158.9, -212.2, -255.6, -305.8, -341.4, -381.6, -416.6, -446.6, -470.8, -495.3])/60
    # percent
    tdv_obs = np.array([-0.058, 0.005, -0.018, -0.009, 0.0020000000000000018, 
                        0.013, -0.031, -0.01, -0.005, -0.005, -0.023999999999999994, 
                        -0.043, -0.043, -0.05, -0.026, -0.047, 0.002, -0.076, -0.043, 
                        -0.017, -0.057, -0.049000000000000016, -0.041, 0.009,
                        0.008499999999999999, 0.008, 0.004, -0.04, -0.028, -0.036, -0.068, 
                        -0.052, -0.052, -0.017, -0.024, 0.052, 0.076, -0.055, -0.128, 0.027, 
                        0.017, 0.053, -0.045, -0.02, -0.018, 0.027, 0.007, 0.008, 0.082, 0.054, 
                        -0.03, 0.023, -0.027, 0.019, 0.039, 0.030000000000000027, 0.02100000000000002,
                         0.012, 0.035, 0.017, 0.057, 0.0, 0.035, -0.004, -0.023, 0.026, 0.063, 0.013, 
                         0.01, -0.017, -0.037, 0.002, -0.067, -0.007, 0.004, 0.004, 0.051, 0.026, 0.045, 
                         -0.047, -0.067, -0.022, 0.001, -0.023, -0.014, -0.044, -0.026999999999999913, -0.01,
                          0.003, -0.048, 0.016, -0.01399999999999979, -0.044, -0.02, -0.015, 0.0029999999999998916, 
                          0.021, 0.044, -0.025, 0.012, 0.007, 0.022, -0.016, 0.029, 0.028, -0.022, 0.016, 0.01933333333333337, 
                          0.02266666666666667, 0.026, 0.05, 0.032, 0.031, 0.006, 0.045, -0.008, -0.006, 0.025, 0.0129999999999999, 
                          0.001, -0.0010000000000000009, -0.003, 0.018, 0.005, 0.01550000000000007, 0.026, -0.018, 0.013, -0.026, 0.013, 
                          -0.009, 0.007, -0.019, 0.027, 0.023])


    transit_time_obs = np.array([55.08028444444445, 66.005793, 76.9315098888889, 87.85153333333335, 98.77308355555554, 109.69463377777777, 
                                120.61021277777776, 131.52759633333332, 142.4415771111111, 153.35861444444444, 164.2711716111111, 175.1837287777778, 
                                186.0944466666667, 197.00405244444448, 207.91476933333334, 218.82402888888888, 229.734468, 240.64261544444446, 
                                251.55520833333333, 262.46377244444443, 273.38087822222224, 284.29996366666666, 295.21904911111113, 306.1411548888889, 
                                317.07517088888886, 328.0091868888889, 338.959627, 349.9080522222221, 360.8728663333334, 371.8373342222223, 
                                382.81853722222223, 393.8021013333334, 404.79559700000004, 415.79193888888886, 426.79876688888885, 437.8139986666667, 
                                448.8245072222223, 459.85154355555557, 470.8601086666667, 481.89658944444454, 492.9045979999999, 503.93698255555546, 
                                514.9387411111112, 525.9641107777778, 536.9565647777778, 547.968878888889, 558.9492485555555, 569.9500358888889, 580.9126972222222, 
                                591.8938307777777, 602.8429514444443, 613.8064461111111, 624.7455657777778, 635.6907975555554, 646.6183894444445, 657.5473474074074, 
                                668.4763053703704, 679.4052633333333, 690.3208413333333, 701.2404481111112, 712.1538733333333, 723.0677152222223, 733.9863497777777, 
                                744.9006083333333, 755.8166724444445, 766.7326681111111, 777.6499822222223, 788.5639630000002, 799.4814170000001, 810.3994255555557,
                                 821.3159757777779, 832.2386381111113, 843.160118888889, 854.0788913333334, 865.0027342222221, 875.924423333333, 886.8470846666668, 
                                 897.7758581111111, 908.6985194444445, 919.6315279999999, 930.5541903333334, 941.490671111111, 952.4306241111111, 963.3786336666667, 
                                 974.3220588888889, 985.2752757777779, 996.2378334444446, 1007.2003911111113, 1018.1593024444445, 1029.1398814444444, 1040.1159455555555, 
                                 1051.1069754444445, 1062.0980053333335, 1073.101638888889, 1084.096591888889, 1095.1049828888888, 1106.1133738888886, 1117.1228407777778, 
                                 1128.1354336666666, 1139.140733888889, 1150.1546452222221, 1161.154252, 1172.1640661111112, 1183.1514496666666, 1194.1531397777778, 
                                 1205.131703888889, 1216.1186013333333, 1227.0773046666668, 1238.036008, 1248.994711333333, 1259.9504292222223, 1270.8823961111113, 
                                 1281.8260991111113, 1292.7426503333334, 1303.6733672222224, 1314.5832507777775, 1325.4994547777776, 1336.407741111111, 1347.3186668333333, 
                                 1358.2295925555557, 1369.1378788888887, 1380.0461652222223, 1390.9552164444442, 1401.8712805555558, 1412.7854701666668, 1423.699659777778,
                                  1434.615723888889, 1445.538663, 1456.456880888889, 1467.385236666667, 1478.3103979999999, 1489.2391714444445, 1500.1714161111113, 
                                  1511.1076885555558, 1522.0437536666668]) 
    transit_time_obs = transit_time_obs/365.25 # day to yr
   
    average_dur = 3.7285/24 # in days
    average_P = 10.9530783 # in days

    tdv_obs = lowess(tdv_obs, range(len(tdv_obs)), frac=0.25)[:, 1] 

    ttv_min = input_scale_dic['ttv_min'] 
    ttv_max = input_scale_dic['ttv_max']
    tdv_min = input_scale_dic['tdv_min']
    tdv_max = input_scale_dic['tdv_max']
    transit_time_min = input_scale_dic['transit_time_min']
    transit_time_max = input_scale_dic['transit_time_max']

    average_dur_min = input_scale_dic['average_dur_min']
    average_dur_max = input_scale_dic['average_dur_max']
    average_P_min = input_scale_dic['average_P_min']
    average_P_max = input_scale_dic['average_P_max']

    average_dur = 2 * (average_dur - average_dur_min)/(average_dur_max - average_dur_min) - 1
    average_P = 2 * (average_P - average_P_min)/(average_P_max - average_P_min) - 1

    if SCALE_INPUT == 'MinMax':

        ttv_obs = 2 * (ttv_obs - ttv_min)/(ttv_max - ttv_min) - 1
        tdv_obs = 2 * (tdv_obs - tdv_min)/(tdv_max - tdv_min) - 1
        transit_time_obs = 2 * (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min) - 1


    elif SCALE_INPUT == 'Minmax01':

        ttv_obs = (ttv_obs - ttv_min)/(ttv_max - ttv_min)
        tdv_obs = (tdv_obs - tdv_min)/(tdv_max - tdv_min)
        transit_time_obs = (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min)
    else:
        tdv_obs *= 100



    ttv_obs = ttv_obs.reshape(1, -1)
    transit_time_obs = transit_time_obs.reshape(1, -1)
    tdv_obs = tdv_obs.reshape(1, -1)


    if NO_TDV:
        input1_obs = np.stack([ttv_obs, transit_time_obs], axis=2)
    else:
        input1_obs = np.stack([ttv_obs, tdv_obs, transit_time_obs], axis=2)

    val1 = torch.from_numpy(input1_obs).float()

    input2_obs = np.stack([average_dur.reshape(-1, 1), average_P.reshape(-1, 1)], axis= 2)
    val2 = torch.from_numpy(input2_obs).float()

    pred_obs = predict(model, 
                        val1, 
                        INCLUDE_APM, 
                        label_pred,
                        device,
                        val2)

    if SCALE_OUTPUT == 'MinMax01':
        pred_obs = pred_obs * (label_max - label_min) + label_min
    elif SCALE_OUTPUT == 'MinMax':
        pred_obs = (pred_obs + 1) / 2 * (label_max - label_min) + label_min
    else:
        pred_obs *= label_max

    print(f'The model prediction for observation is {pred_obs} at the end of current epoch (inter).')
    return pred_obs

def predict_from_obs_GRIT(input_scale_dic,
                     model,
                     INCLUDE_APM,
                     label_pred,
                     device,
                     SCALE_INPUT,
                     SCALE_OUTPUT,
                     label_max,
                     label_min,
                     NO_TDV
                    ):
    ################ Predcit for the observation ##################
    # mins to hr
    ttv_obs = np.array([583.47679047,  539.27488332,  496.74379878,  447.34695048, 401.41949783,  348.06958045,  298.64591889,  242.15884832,
                        188.94536405,  129.92797046,   72.74325193,   11.88396001, -49.09164788, -110.74867831, -174.68353622, -235.5043758 ,
                        -300.67580919, -358.26207986, -421.90351924, -473.05385612, -531.32182533, -572.19417149, -620.3422636 , -646.86614707,
                        -679.68378321, -688.2161838 , -700.79603518, -688.8763428 , -677.56547251, -644.56246057, -607.83972332, -555.24141734,
                        -494.16458225, -425.45794082, -343.52104982, -263.75579994, -166.33791448,  -81.54322698,   24.74593051,  108.11497281,
                        216.06691091,  291.67842045,  393.96660622,  456.15931152, 545.86250237,  590.27453685,  661.51850552,  685.65497881,
                        734.36597538,  737.9496079 ,  762.51006085,  747.352488,
                        748.75201643,  718.22528538,  699.58302394,  657.79952625,
                        623.48875647,  574.52107041,  529.20912566,  476.53455886,
                        424.47869217,  370.67341099,  315.29988785,  261.99203899,
                        205.78369668,  153.7692629 ,   98.34972896,   47.79616081,
                        -5.86877605,  -55.14695146, -106.48321796, -154.75475768,
                        -203.31078929, -250.66598228, -295.85458764, -341.9635308 ,
                        -382.85172863, -426.78277316, -461.95206586, -502.05887919,
                        -529.58549629, -563.52257024, -581.05454958, -605.9248782 ,
                        -610.98368713, -623.62103771, -614.04442882, -611.51022513,
                        -585.94516542, -566.08422648, -524.46991792, -486.46516393,
                        -430.27256932, -374.97983554, -307.22977504, -237.21639044,
                        -162.24631621,  -81.58418678,   -4.66823737,   81.4529093 ,
                        154.48471014,  240.16636475,  303.4385382 ,  382.47890216,
                        430.62455197,  497.00249012,  525.82891779,  574.30640426,
                        581.48127703,  608.38425399,  593.88689082,  597.81318906,
                        563.92951734,  546.06783393,  496.82962721,  460.6028284 ,
                        400.94520408,  351.16001907,  286.04677223,  227.92685363,
                        161.66938226,  100.04154566,   35.96077589,  -25.22279712,
                        -84.88023811, -142.71814788, -196.70071196, -249.28798224,
                        -297.11483361, -343.34251203, -385.05759707, -424.38374936,
                        -460.2781904 , -492.47955864, -522.86476599])/60
    # percent
    tdv_obs = np.array([0.01247038,  0.00939397,  0.00852441,  0.00558716,  0.00458765,
                        0.00188986,  0.00074017, -0.00162519, -0.00292018, -0.00488855,
                        -0.00628564, -0.00785976, -0.00937894, -0.01050334, -0.01206708,
                        -0.01275447, -0.0143603 , -0.01458768, -0.01621948, -0.0159929 ,
                        -0.017507  , -0.01689349, -0.01815339, -0.01726722, -0.01815517,
                        -0.01706411, -0.01735368, -0.01616325, -0.01581214, -0.01463075,
                        -0.01355717, -0.01243316, -0.0105745 , -0.00949073, -0.00697225,
                        -0.00594306, -0.00278175, -0.00182333,  0.00188045,  0.0027503 ,
                        0.00686433,  0.00765119,  0.01194507,  0.01270711,  0.01695   ,
                        0.01758857,  0.02143851,  0.02196418,  0.02504071,  0.02548823,
                        0.02748131,  0.02776442,  0.0286059 ,  0.02874406,  0.02847049,
                        0.02842883,  0.02717103,  0.02694286,  0.02492673,  0.02455849,
                        0.0220228 ,  0.02152188,  0.01868628,  0.01800302,  0.01506718,
                        0.01422723,  0.01135852,  0.01037777,  0.00762195,  0.00654434,
                        0.0040572 ,  0.0027934 ,  0.00063373, -0.00071019, -0.00249974,
                        -0.00398456, -0.00532161, -0.00688875, -0.0077712 , -0.00946452,
                        -0.00986423, -0.01164586, -0.01158629, -0.01338941, -0.01290238,
                        -0.01468622, -0.01382283, -0.01547276, -0.01436299, -0.01576573,
                        -0.0145277 , -0.01561212, -0.0143209 , -0.0150104 , -0.01375083,
                        -0.01403865, -0.01280777, -0.01262974, -0.01147068, -0.01083531,
                        -0.00972074, -0.00853674, -0.00752486, -0.00575506, -0.00488244,
                        -0.00258585, -0.00183468,  0.00088779,  0.00154593,  0.00435641,
                        0.00486322,  0.00737856,  0.0077346 ,  0.00957982,  0.00978634,
                        0.01073221,  0.01076032,  0.01073903,  0.01062154,  0.00963705,
                        0.00938523,  0.00761098,  0.00719655,  0.0048282 ,  0.00429106,
                        0.0015255 ,  0.0008181 , -0.00217746, -0.00294043, -0.00600677,
                        -0.00693892, -0.00997679, -0.0109953 , -0.01379522, -0.01493329])


    transit_time_obs = np.array([10.96285231,   21.92184257,   32.88199313,   43.83737579,
                                54.79516776,   65.74780524,   76.70316928,   87.65362818,
                                98.6063604 ,  109.55506213,  120.50503655,  131.45245919,
                                142.39980105,  153.3466697 ,  164.29195652,  175.23940586,
                                186.18383395,  197.13352951,  208.0790201 ,  219.03318507,
                                229.98240723,  240.94370969,  251.89995954,  262.87122621,
                                273.83812222,  284.82188297,  295.80283299,  306.80079659,
                                317.79833739,  328.81094218,  339.82613011,  350.85234274,
                                361.88444324,  372.92184222,  383.96842887,  395.01350744,
                                406.07084478,  417.11941601,  428.18291395,  439.23049515,
                                450.29514781,  461.33734183,  472.39806132,  483.43093673,
                                494.48291665,  505.50344437,  516.54260538,  527.54905285,
                                538.57256596,  549.56474063,  560.57148253,  571.55064247,
                                582.54130039,  593.50978731,  604.48652732,  615.44719704,
                                626.41305614,  637.36873684,  648.32695624,  659.28006271,
                                670.23359883,  681.18592009,  692.13715228,  703.08981897,
                                714.04047143,  724.99403633,  735.94523657,  746.89981596,
                                757.85223467,  768.80769975,  779.7617356 ,  790.71789972,
                                801.67386629,  812.63066677,  823.58897183,  834.54663776,
                                845.50792921,  856.46710757,  867.43237048,  878.39420456,
                                889.36477488,  900.3308935 ,  911.30840455,  922.28081951,
                                933.26699248,  944.24790258,  955.24423903,  966.23568493,
                                977.24312447,  988.24660282,  999.26518768, 1010.2812659 ,
                                1021.30997457, 1032.33805833, 1043.37479301, 1054.41309945,
                                1065.45484803, 1076.50054943, 1087.54364932, 1098.5931417 ,
                                1109.63354426, 1120.68273144, 1131.71635648, 1142.76093165,
                                1153.78405216, 1164.81983398, 1175.82953837, 1186.85288932,
                                1197.8475579 , 1208.85592656, 1219.83554497, 1230.8279576 ,
                                1241.7941133 , 1252.77139539, 1263.72688822, 1274.69141675,
                                1285.63967388, 1296.59478686, 1307.53925536, 1318.48858034,
                                1329.43225424, 1340.37914316, 1351.32432866, 1362.27152609,
                                1373.21978335, 1384.16930416, 1395.1215023 , 1406.0746694 ,
                                1417.03114234, 1427.98872581, 1438.94944303, 1449.91181924,
                                1460.87657857, 1471.84390254, 1482.81248774]) 
    transit_time_obs = transit_time_obs/365.25 # day to yr
   
    average_dur = 0.1384122289753376 # in days 
    average_P = 10.9896860

    ttv_min = input_scale_dic['ttv_min'] 
    ttv_max = input_scale_dic['ttv_max']
    tdv_min = input_scale_dic['tdv_min']
    tdv_max = input_scale_dic['tdv_max']
    transit_time_min = input_scale_dic['transit_time_min']
    transit_time_max = input_scale_dic['transit_time_max']

    average_dur_min = input_scale_dic['average_dur_min']
    average_dur_max = input_scale_dic['average_dur_max']
    average_P_min = input_scale_dic['average_P_min']
    average_P_max = input_scale_dic['average_P_max']

    average_dur = 2 * (average_dur - average_dur_min)/(average_dur_max - average_dur_min) - 1
    average_P = 2 * (average_P - average_P_min)/(average_P_max - average_P_min) - 1

    if SCALE_INPUT == 'MinMax':

        ttv_obs = 2 * (ttv_obs - ttv_min)/(ttv_max - ttv_min) - 1
        tdv_obs = 2 * (tdv_obs - tdv_min)/(tdv_max - tdv_min) - 1
        transit_time_obs = 2 * (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min) - 1


    elif SCALE_INPUT == 'Minmax01':

        ttv_obs = (ttv_obs - ttv_min)/(ttv_max - ttv_min)
        tdv_obs = (tdv_obs - tdv_min)/(tdv_max - tdv_min)
        transit_time_obs = (transit_time_obs - transit_time_min)/(transit_time_max - transit_time_min)
    else:
        tdv_obs *= 100



    ttv_obs = ttv_obs.reshape(1, -1)
    transit_time_obs = transit_time_obs.reshape(1, -1)
    tdv_obs = tdv_obs.reshape(1, -1)


    if NO_TDV:
        input1_obs = np.stack([ttv_obs, transit_time_obs], axis=2)
    else:
        input1_obs = np.stack([ttv_obs, tdv_obs, transit_time_obs], axis=2)

    val1 = torch.from_numpy(input1_obs).float()

    input2_obs = np.stack([average_dur.reshape(-1, 1), average_P.reshape(-1, 1)], axis= 2)
    val2 = torch.from_numpy(input2_obs).float()

    pred_obs = predict(model, 
                        val1, 
                        INCLUDE_APM, 
                        label_pred,
                        device,
                        val2)

    if SCALE_OUTPUT == 'MinMax01':
        pred_obs = pred_obs * (label_max - label_min) + label_min
    elif SCALE_OUTPUT == 'MinMax':
        pred_obs = (pred_obs + 1) / 2 * (label_max - label_min) + label_min
    else:
        pred_obs *= label_max

    print(f'The model prediction for observation is {pred_obs} at the end of current epoch (GRIT).')
    return pred_obs