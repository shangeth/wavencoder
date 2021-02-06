import torch
import torch.nn as nn
import torch.functional as F

class CNN1d(nn.Module):
    def __init__(self, conv_layers):
        super(CNN1d, self).__init__()
        self.conv_layers = nn.ModuleList()

        in_d = 1
        for dim, k, s in conv_layers:
            self.conv_layers.append(self.cnn_block(in_d, dim, k, s))
            in_d = dim

    def cnn_block(self, n_in, n_out, k, s):
        block = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=k, stride=s),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU()
        )
        return block

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x


