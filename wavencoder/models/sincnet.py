import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import os
import urllib.request
from tqdm import tqdm

from wavencoder.layers import SincConvLayer
from wavencoder.utils import _reporthook

def act_fun(act_type):

 if act_type=="relu":
    return nn.ReLU()
            
 if act_type=="tanh":
    return nn.Tanh()
            
 if act_type=="sigmoid":
    return nn.Sigmoid()
           
 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)
            
 if act_type=="elu":
    return nn.ELU()
                     
 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)
        
 if act_type=="linear":
    return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
                      
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

class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()
        
        self.input_dim=int(options['input_dim'])
        self.fc_lay=options['fc_lay']
        self.fc_drop=options['fc_drop']
        self.fc_use_batchnorm=options['fc_use_batchnorm']
        self.fc_use_laynorm=options['fc_use_laynorm']
        self.fc_use_laynorm_inp=options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp=options['fc_use_batchnorm_inp']
        self.fc_act=options['fc_act']
        
       
        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
       

       
        # input layer normalization
        if self.fc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
          
        # input batch normalization    
        if self.fc_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
           
        self.N_fc_lay=len(self.fc_lay)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_fc_lay):
            
         # dropout
         self.drop.append(nn.Dropout(p=self.fc_drop[i]))
         
         # activation
         self.act.append(act_fun(self.fc_act[i]))
         
         
         add_bias=True
         
         # layer norm initialization
         self.ln.append(LayerNorm(self.fc_lay[i]))
         self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
         
         if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
             add_bias=False
         
              
         # Linear operations
         self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
         
         # weight initialization
         self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
         self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
         
         current_input=self.fc_lay[i]
         
         
    def forward(self, x):
        
      # Applying Layer/Batch Norm
      if bool(self.fc_use_laynorm_inp):
        x=self.ln0((x))
        
      if bool(self.fc_use_batchnorm_inp):
        x=self.bn0((x))
        
      for i in range(self.N_fc_lay):

        if self.fc_act[i]!='linear':
            
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.act[i](self.wx[i](x)))
           
        else:
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.ln[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.bn[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.wx[i](x)) 
          
      return x

class SincNetModel(nn.Module):
    
    def __init__(self,options):
       super(SincNetModel,self).__init__()
    
       self.cnn_N_filt=options['cnn_N_filt']
       self.cnn_len_filt=options['cnn_len_filt']
       self.cnn_max_pool_len=options['cnn_max_pool_len']
       
       
       self.cnn_act=options['cnn_act']
       self.cnn_drop=options['cnn_drop']
       
       self.cnn_use_laynorm=options['cnn_use_laynorm']
       self.cnn_use_batchnorm=options['cnn_use_batchnorm']
       self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
       self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
       
       self.input_dim=int(options['input_dim'])
       
       self.fs=options['fs']
       
       self.N_cnn_lay=len(options['cnn_N_filt'])
       self.conv  = nn.ModuleList([])
       self.bn  = nn.ModuleList([])
       self.ln  = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])
       
             
       if self.cnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
           
       if self.cnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
       current_input=self.input_dim 
       
       for i in range(self.N_cnn_lay):
         
         N_filt=int(self.cnn_N_filt[i])
         len_filt=int(self.cnn_len_filt[i])
         
         # dropout
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
         
         # activation
         self.act.append(act_fun(self.cnn_act[i]))
                    
         # layer norm initialization         
         self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

         self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            

         if i==0:
          self.conv.append(SincConvLayer(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs))
              
         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
          
         current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

         
       self.out_dim=current_input*N_filt



    def forward(self, x):
       batch=x.shape[0]
       seq_len=x.shape[1]
       
       if bool(self.cnn_use_laynorm_inp):
        x=self.ln0((x))
        
       if bool(self.cnn_use_batchnorm_inp):
        x=self.bn0((x))
        
       x=x.view(batch,1,seq_len)
       
       for i in range(self.N_cnn_lay):
           
         if self.cnn_use_laynorm[i]:
          if i==0:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
          else:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
          
         if self.cnn_use_batchnorm[i]:
          x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

         if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
          x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

       x = x.view(batch,-1)
       return x
   
class SincNet(nn.Module):
    def __init__(self, only_cnn=False, pretrained=True, pretrained_path=None,  device=torch.device("cpu")):
        super(SincNet, self).__init__()
        self.only_cnn = only_cnn
        self.device = device
        
        # [cnn]
        fs = 16000
        cw_len=200
        wlen=int(fs*cw_len/1000.00)
        cnn_N_filt=80,60,60
        cnn_len_filt=251,5,5
        cnn_max_pool_len=3,3,3
        cnn_use_laynorm_inp=True
        cnn_use_batchnorm_inp=False
        cnn_use_laynorm=True,True,True
        cnn_use_batchnorm=False,False,False
        cnn_act=['relu','relu','relu']
        cnn_drop=0.0,0.0,0.0

        # [dnn]
        fc_lay=2048,2048,2048
        fc_drop=[0.0,0.0,0.0]
        fc_use_laynorm_inp=True
        fc_use_batchnorm_inp=False
        fc_use_batchnorm=True,True,True
        fc_use_laynorm=False,False,False
        fc_act=['leaky_relu','leaky_relu','leaky_relu']

        # [class]
        class_lay=[10]
        class_drop=[0.0]
        class_use_laynorm_inp=False
        class_use_batchnorm_inp=False
        class_use_batchnorm=[False]
        class_use_laynorm=[False]
        class_act=['softmax']

        CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

        self.cnn_network = SincNetModel(CNN_arch)

        DNN1_arch = {'input_dim': self.cnn_network.out_dim,
                'fc_lay': fc_lay,
                'fc_drop': fc_drop, 
                'fc_use_batchnorm': fc_use_batchnorm,
                'fc_use_laynorm': fc_use_laynorm,
                'fc_use_laynorm_inp': fc_use_laynorm_inp,
                'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
                'fc_act': fc_act,
                }

        DNN2_arch = {'input_dim':fc_lay[-1] ,
                'fc_lay': class_lay,
                'fc_drop': class_drop, 
                'fc_use_batchnorm': class_use_batchnorm,
                'fc_use_laynorm': class_use_laynorm,
                'fc_use_laynorm_inp': class_use_laynorm_inp,
                'fc_use_batchnorm_inp':class_use_batchnorm_inp,
                'fc_act': class_act,
                }
        if not self.only_cnn:
            self.ann_network1 = MLP(DNN1_arch)
            self.ann_network2 = MLP(DNN2_arch)
        
        if pretrained:
            filename = "model_raw.pkl"
            pretrained_weights_link = "https://bitbucket.org/mravanelli/sincnet_models/raw/f3588bdf02b4634a975549500f6f68bd1a80a997/SincNet_TIMIT/model_raw.pkl"
            if pretrained_path == None:
                if not os.path.exists(filename):
                    print(f'Downloading the pretrained weights from sincnet({pretrained_weights_link}) ...', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(pretrained_weights_link, filename, reporthook=_reporthook(t))
                cp = torch.load(filename, map_location=self.device) 
            else: 
                cp = torch.load(pretrained_path, map_location=self.device)
            self.cnn_network.load_state_dict(cp['CNN_model_par'])
            if not self.only_cnn:
                self.ann_network1.load_state_dict(cp['DNN1_model_par'])
                # self.ann_network2.load_state_dict(cp['DNN2_model_par'])

    def forward(self, x):
        x = x.squeeze(1)
        y = self.cnn_network(x)
        if not self.only_cnn:
            y = self.ann_network1(y)
            # y = self.ann_network2(y)
        return y