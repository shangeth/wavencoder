import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftAttention(nn.Module):
    '''
    https://arxiv.org/abs/1803.10916
    '''
    def __init__(self, emb_dim, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, self.attn_dim)
        self.v = nn.Parameter(torch.Tensor(self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)

        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, values):
        attention_weights = self._get_weights(values)
        values = values.transpose(1,2)
        weighted = torch.mul(values, attention_weights.unsqueeze(1).expand_as(values))
        representations = weighted.sum(2).squeeze()
        return representations

    def _get_weights(self, values):
        batch_size = values.size(0)
        weights = self.W(values)
        weights = torch.tanh(weights)
        e = weights @ self.v
        attention_weights = torch.softmax(e.squeeze(1), dim=-1)
        return attention_weights


class AdditiveAttention(nn.Module):
    '''
    https://arxiv.org/abs/1409.0473
    AKA  Bahdanau attention
    '''
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim	 	 
        self.decoder_dim = decoder_dim	 	 
        self.v = torch.nn.Parameter(torch.rand(self.decoder_dim))	 	 
        self.W1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)	 	 
        self.W2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def forward(self, values, query):
        weights = self._get_weights(values, query)
        weights = F.softmax(weights, 1)
        return torch.mul(weights.unsqueeze(2).repeat(1, 1, values.size(2)), values).sum(1)

    def _get_weights(self, values, query): 
        query = query.unsqueeze(1).repeat(1, values.size(1), 1)
        weights = self.W1(query) + self.W2(values)
        return torch.tanh(weights) @ self.v
    

class MultiplicativeAttention(nn.Module):
    '''
    https://arxiv.org/abs/1409.0473
    AKA  Bahdanau attention
    '''
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim	 	 
        self.decoder_dim = decoder_dim	 	 	 	 
        self.W = torch.nn.Parameter(torch.rand(self.decoder_dim, self.encoder_dim))	 	 	 

    def forward(self, values, query):
        weights = self._get_weights(values, query)
        weights = F.softmax(weights, 1)
        return torch.mul(weights.unsqueeze(2).repeat(1, 1, values.size(2)), values).sum(1)

    def _get_weights(self, values, query): 
        weights = (query @ self.W).unsqueeze(1) @ values.transpose(1, 2)
        return weights.squeeze(1)/np.sqrt(self.decoder_dim)


class DotAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    
    def forward(self, values, query):
        attention_weights = self._get_weights(values, query)
        representations = torch.bmm(values.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return representations

    def _get_weights(self, values, query):
        hidden = query.squeeze(0)
        attention_weights = torch.bmm(values, hidden.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        return attention_weights


if __name__ == "__main__":

    # Soft Attention
    values = torch.randn(16, 54, 128) # [Batch, seq, dim]
    attn_layer = SoftAttention(128, 256)
    attn_repr = attn_layer(values)
    attn_weights = attn_layer._get_weights(values)
    print(attn_repr.shape, attn_weights.shape) 

    # Dot Attention
    values = torch.randn(16, 54, 128) # [Batch, seq, dim]
    query = torch.randn(16, 128) # [batch, dim]
    attn_layer = DotAttention()
    attn_repr = attn_layer(values, query)
    attn_weights = attn_layer._get_weights(values, query)
    print(attn_repr.shape, attn_weights.shape) 

    # Additive Attention
    values = torch.randn(16, 54, 128) # [Batch, seq, encoder_dim]
    query = torch.randn(16, 256) # [Batch, decoder_dim]
    attn_layer = AdditiveAttention(128, 256)
    attn_repr = attn_layer(values, query)
    attn_weights = attn_layer._get_weights(values, query)
    print(attn_repr.shape, attn_weights.shape)

    # Multiplicative Attention
    values = torch.randn(16, 54, 128) # [Batch, seq, encoder_dim]
    query = torch.randn(16, 256) # [Batch, decoder_dim]
    attn_layer = MultiplicativeAttention(128, 256)
    attn_repr = attn_layer(values, query)
    attn_weights = attn_layer._get_weights(values, query)
    print(attn_repr.shape, attn_weights.shape)

