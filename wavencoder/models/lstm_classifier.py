import torch
import torch.nn as nn
import torch.nn.functional as F
from wavencoder.layers import DotAttention, SoftAttention

class LSTM_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes):
        super(LSTM_Classifier, self).__init__()
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x.transpose(1,2))
        lstm_out = lstm_out[:, -1, :]
        out = self.classifier(lstm_out)
        return out

class LSTM_Attn_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes, return_attn_weights=False, attn_type='dot'):
        super(LSTM_Attn_Classifier, self).__init__()
        self.return_attn_weights = return_attn_weights
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.attn_type = attn_type

        if self.attn_type == 'dot':
            self.attention = DotAttention()
        elif self.attn_type == 'soft':
            self.attention = SoftAttention(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, n_classes)


    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x.transpose(1,2))

        if self.attn_type == 'dot':
            attn_output = self.attention(lstm_out, hidden)
            attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == 'soft':
            attn_output = self.attention(lstm_out)
            attn_weights = self.attention._get_weights(lstm_out)

        out = self.classifier(attn_output)
        if self.return_attn_weights:
            return out, attn_weights
        else:
            return out