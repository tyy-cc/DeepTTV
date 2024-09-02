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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransAm(nn.Module):
    def __init__(self, 
                 n_layers_gru, 
                 n_layers_att, 
                 feature_size, 
                 NHEAD, 
                 dropout=0.1, 
                 has_second_input=True, 
                 second_input_size=2, 
                 num_labels=1, 
                 act_func='ReLu', 
                 input_size=3, 
                 n_layers_deco=2, 
                 multi_deco=False, 
                 deco_list = ['mass', 'e', 'incli', 'omega', 'Omega', ' mean_ano', 'a'],
                 init_linear_w='default',
                 init_rnn_w='default',
                 use_token=True,
                 norm_type='None'):
        super(TransAm, self).__init__()

        self.model_type = 'Transformer'
        self.has_second_input = has_second_input
        self.src_mask = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_deco = multi_deco
        self.feature_size = feature_size
        self.use_token = use_token
        self.norm_type = norm_type

        # Norm layers
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        if self.has_second_input:
            self.layer_norm2 = nn.LayerNorm(second_input_size)
            self.batch_norm2 = nn.BatchNorm1d(second_input_size)
        #LSTM/GRU
        self.gru_b = nn.GRU(input_size = input_size, hidden_size = feature_size, num_layers=n_layers_gru, batch_first=True)
        
        # if has second input
        if self.has_second_input:
            self.encode_input1 = nn.Linear(second_input_size, feature_size//2)
            self.encode_lrelu1 = nn.LeakyReLU()
            self.encode_input2 = nn.Linear(feature_size//2, feature_size)
            self.encode_lrelu2 = nn.LeakyReLU()

        # Attention
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=feature_size, nhead=NHEAD, dropout=dropout, batch_first=True)
        self.transformer_encoder3 = nn.TransformerEncoder(self.encoder_layer3, num_layers=n_layers_att)

        # Decoder
        if not multi_deco: # only one decoder
            self.decoder = nn.ModuleList()
            self.decoder_act = nn.ModuleList()

            feature_size_before = feature_size
            if n_layers_deco > 0:

                for _ in range(n_layers_deco):
                    self.decoder.append(nn.Linear(feature_size_before, feature_size_before//2))
                    feature_size_before = feature_size_before // 2

                    if act_func == 'ReLu':
                        self.decoder_act.append(nn.LeakyReLU())
                    else:
                        self.decoder_act.append(nn.Tanh())


            self.decoder.append(nn.Linear(feature_size_before, num_labels))
        else: # multi decoders
            '''
            mass_decoder: 'm_t', 'm_nont'
            e_decoder: 'e_t', 'e_nont'
            incli_decoder:  'cos(incli_t)', 'cos(incli_nont)'
            omega_decoder: 'cos(omega_t)', 'sin(omega_t)', 'cos(omega_nont)', 'sin(omega_nont)'
            Ome_decoder: 'cos(Omega_t)', 'sin(Omega_t)', 'cos(Omega_nont)', 'sin(Omega_nont)'
            Mean_ano_decoder:'cos(Mean_ano_nont)', 'sin(Mean_ano_nont)'
            a_decoder: 'a_nont'

            '''
            self.multi_deco_list = nn.ModuleList()
            self.multi_deco_act_list = nn.ModuleList()
            feature_size_before = feature_size       

            if n_layers_deco > 0:
                for _ in range(n_layers_deco): # loop layers
                    temp_multi_deco_layer = nn.ModuleList()
                    temp_multi_deco_act = nn.ModuleList()
                    for _ in range(len(deco_list)): # loop different decoders
                        temp_multi_deco_layer.append(nn.Linear(feature_size_before, feature_size_before//2))

                        if act_func == 'ReLu':
                            temp_multi_deco_act.append(nn.LeakyReLU())
                        else:
                            temp_multi_deco_act.append(nn.Tanh())  

                    feature_size_before //= 2

                    self.multi_deco_list.append(temp_multi_deco_layer)
                    self.multi_deco_act_list.append(temp_multi_deco_act)  
            
            multi_deco_output_layer = nn.ModuleList()

            # loop different decoders for the last layer
            # ['mass', 'e', 'incli', 'omega', 'Omega', ' mean_ano', 'a']
            for iii in range(len(deco_list)):
                if deco_list[iii] == 'mass' or deco_list[iii] == 'e':
                    n_labels = 2 
                elif deco_list[iii] == 'incli' or deco_list[iii] == 'omega' or deco_list[iii] == 'Omega' or deco_list[iii] == 'mean_ano':
                    n_labels = 4 
                else:
                    n_labels = 1
                multi_deco_output_layer.append(nn.Linear(feature_size_before, n_labels))

            self.multi_deco_list.append(multi_deco_output_layer)

        self.init_weights(init_linear_w, init_rnn_w)

        #self.pe = nn.Parameters(torch.zeros(1, 20, feature_size), Trainable=pos_encode) # if you want trainable parameters

    def init_weights(self, init_linear_w, init_rnn_w):

        if init_linear_w == 'xav_uni':
            if self.has_second_input:
                nn.init.xavier_uniform_(self.encode_input1.weight)
                nn.init.xavier_uniform_(self.encode_input2.weight)
                self.encode_input1.bias.data.zero_()
                self.encode_input2.bias.data.zero_()
            
            if not self.multi_deco:
                for layer in self.decoder:
                    nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.zero_()
            else:
                for layer in self.multi_deco_list:
                    for deco_this_layer in layer:
                        nn.init.xavier_uniform_(deco_this_layer.weight)
                        deco_this_layer.bias.data.zero_()

        if init_rnn_w == 'ortho':
            for name, param in self.gru_b.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)

    def forward(self, src1, src2=None):

        batchsize = src1.shape[0]

        token = torch.zeros(batchsize, 1, self.feature_size)

        if self.norm_type == 'Layer':
            src1 = self.layer_norm1(src1) 
            if self.has_second_input:
                src2 = self.layer_norm2(src2)
        elif self.norm_type == 'Batch':
            src1 = src1.permute(0, 2, 1) # [N, L, F] -> [N, F, L]
            src1 = self.batch_norm1(src1)
            src1 = src1.permute(0, 2, 1) # [N, F, L] -> [N, L, F]
            if self.has_second_input:
                src2 = src2.permute(0, 2, 1)
                src2 = self.batch_norm2(src2)
                src2 = src2.permute(0, 2, 1)
            
        output1, h_n_b = self.gru_b(src1, None)

# combine two seq
        if self.has_second_input:

            output2 = self.encode_input1(src2)
            output2 = self.encode_lrelu1(output2)
            output2 = self.encode_input2(output2)
            output2 = self.encode_lrelu2(output2)
            output = torch.cat((output1, output2), 1) #[batch, seq_len_plus, feature_size]
        else:
            output = output1
        # output += self.pe
        if self.use_token:
            output = torch.cat((output, token.to(self.device)), 1)

        output = self.transformer_encoder3(output)
        output = output[:, -1, :]

        if not self.multi_deco:
            for ii in range(len(self.decoder) - 1):
                ff_layer = self.decoder[ii]
                output = ff_layer(output)
                act_fun = self.decoder_act[ii]
                output = act_fun(output)
            # last output layer
            ff_layer = self.decoder[-1]
            output = ff_layer(output)

            return output
        else:
            
            for jj in range(7): # 7 decoders
                for ii in range(len(self.multi_deco_list) - 1):

                    ff_layer = self.multi_deco_list[ii][jj]
                    act_fun = self.multi_deco_act_list[ii][jj]
                    if ii == 0: # if first layer
                        temp_output = ff_layer(output) 
                        temp_output = act_fun(temp_output)
                    else:
                        temp_output = ff_layer(temp_output)
                        temp_output = act_fun(temp_output)
                
                # if zero decoder layer, init temp_output
                if len(self.multi_deco_list) == 1:
                    temp_output = output
                # last output layer
                ff_layer = self.multi_deco_list[-1][jj]
                temp_output = ff_layer(temp_output)
                #print(ff_layer)
                if jj == 0:
                    multi_output = temp_output 
                    #print(f'temp multi decoder output shape is {temp_output.shape}')
                else:
                    #print(f'temp multi decoder output shape is {multi_output.shape}')
                    multi_output = torch.cat((multi_output, temp_output), 1)
            return multi_output