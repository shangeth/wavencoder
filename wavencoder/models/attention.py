import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftAttention(nn.Module):
    def __init__(self, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.W = nn.Parameter(torch.Tensor(self.attn_dim, self.attn_dim), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(1, self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)
        for weight in self.W:
            nn.init.uniform_(weight, -stdv, stdv)
        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs):
        inputs = inputs.transpose(1,2)
        batch_size = inputs.size(0)
        weights = torch.bmm(self.W.unsqueeze(0).repeat(batch_size, 1, 1), inputs)
        e = torch.tanh(weights)

        e = torch.bmm(self.v.unsqueeze(0).repeat(batch_size, 1, 1), e)
        attentions = torch.softmax(e.squeeze(1), dim=-1)
        weighted = torch.mul(inputs, attentions.unsqueeze(1).expand_as(inputs))
        representations = weighted.sum(2).squeeze()
        return representations, attentions


class DotAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    
    def forward(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        attentions = F.softmax(attn_weights, 1)
        representations = torch.bmm(lstm_output.transpose(1, 2), attentions.unsqueeze(2)).squeeze(2)
        return representations, attentions