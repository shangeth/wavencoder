import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import os
import urllib.request
from tqdm import tqdm
from wavencoder.layers import SincConvLayer
from wavencoder.utils import _reporthook


class FRM(nn.Module):
    def __init__(self, nb_dim, do_add = True, do_mul = True):
        super(FRM, self).__init__()
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul
    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        if self.do_mul: x = x * y
        if self.do_add: x = x + y
        return x

class Residual_block_wFRM(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block_wFRM, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
            out_channels = nb_filts[1],
            kernel_size = 3,
            padding = 1,
            stride = 1)
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
            out_channels = nb_filts[1],
            padding = 1,
            kernel_size = 3,
            stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
                out_channels = nb_filts[1],
                padding = 0,
                kernel_size = 1,
                stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        self.frm = FRM(
            nb_dim = nb_filts[1],
            do_add = True,
            do_mul = True)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        out = self.frm(out)
        return out

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RawNet2(nn.Module):
    def __init__(self, d_args, return_code=True):
        super(RawNet2, self).__init__()
        self.return_code = return_code

        self.ln = LayerNorm(d_args['nb_samp'])
        self.first_conv = SincConvLayer(in_channels = d_args['in_channels'],
            out_channels = d_args['filts'][0],
            kernel_size = d_args['first_conv']
            )

        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)
        
        self.block0 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][1], first = True))
        self.block1 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][1]))
 
        self.block2 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block_wFRM(nb_filts = d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
            hidden_size = d_args['gru_node'],
            num_layers = d_args['nb_gru_layer'],
            batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
            out_features = d_args['nb_fc_node'])

        if not self.return_code:
            self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
                out_features = d_args['nb_classes'],
                bias = True)
            
        
    def forward(self, x, y = 0, return_code=False):
        #follow sincNet recipe
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x=x.view(nb_samp,1,len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)
        
        x = self.block0(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        code = self.fc1_gru(x)
        if self.return_code: 
            return code
        else:
            code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
            code = torch.div(code, code_norm)
            out = self.fc2_gru(code)
            return out


class RawNet2Model(nn.Module):
    def __init__(self, pretrained=True, pretrained_path=None, device=torch.device("cpu"), return_code=True, class_dim=10):
        super().__init__()
        self.device = device
        self.return_code = return_code
        
        d_args = {
        'nb_classes':class_dim, 'first_conv': 251, 'in_channels': 1, 'filts': [128, [128, 128], [128, 256], [256, 256]], 
        'blocks': [2, 4], 'nb_fc_att_node': [1], 'nb_fc_node': 1024, 'gru_node': 1024, 'nb_gru_layer': 1, 'nb_samp': 59049, 
        'model': {
            'first_conv': 251, 'in_channels': 1, 'filts': [128, [128, 128], [128, 256], [256, 256]], 
            'blocks': [2, 4], 'nb_fc_att_node': [1], 'nb_fc_node': 1024, 'gru_node': 1024, 'nb_gru_layer': 1, 'nb_samp': 59049
            }
        }
   

        self.encoder = RawNet2(d_args, return_code=return_code)


        if pretrained:
            filename = "rawnet2_best_weights.pt"
            pretrained_weights_link = "https://github.com/Jungjee/RawNet/raw/master/Pre-trained_model/rawnet2_best_weights.pt"
            if pretrained_path == None:
                if not os.path.exists(filename):
                    print(f'Downloading the pretrained weights from Jungjee/RawNet({pretrained_weights_link}) ...', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(pretrained_weights_link, filename, reporthook=_reporthook(t))
                cp = torch.load(filename, map_location=self.device)
            else: 
                cp = torch.load(pretrained_path, map_location=self.device)
            pretrained_dict = cp
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)

    def forward(self, x):
        x = x.squeeze(1)
        return self.encoder(x, return_code=self.return_code)