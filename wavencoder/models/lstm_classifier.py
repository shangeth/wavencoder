import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, inp_size, hidden_size, n_classes):
        super(LSTM_Attn_Classifier, self).__init__()
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output, soft_attn_weights = self.attention_net(lstm_out, hidden)
        out = self.classifier(attn_output)
        return out, soft_attn_weights