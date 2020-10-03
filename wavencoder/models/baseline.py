import torch
import torch.nn as nn
import torch.functional as F

class CNN1d(nn.Module):
    def __init__(self, conv_layers):
        super(CNN1d, self).__init__()

    def cnn_block(n_in, n_out, k, s):
        block = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=k, stride=s),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU()
        )
        return block

    
