from wavencoder.models import CNN1d

# from wavencoder.models import Wav2Vec
# from wavencoder.models import SincNet
from wavencoder.models import LSTM_Classifier, LSTM_Attn_Classifier
import torch
import torch.nn as nn


conv_layers = [(16, 10, 5), (32, 10, 5), (64, 10, 5)]
encoder = CNN1d(conv_layers)
classifier = LSTM_Attn_Classifier(64, 64, 2)

x = torch.randn(2, 1, 16000) 

z = encoder(x)
y = classifier(z)
print(z.shape, y.shape)