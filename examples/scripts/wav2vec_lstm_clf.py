from wavencoder.models import Wav2Vec
from wavencoder.models import SincNet
from wavencoder.models import LSTM_Classifier, LSTM_Attn_Classifier
import torch
import torch.nn as nn

# encoder = Wav2Vec(pretrained=False)
# classifier = LSTM_Attn_Classifier(512, 64, 2)
# x = torch.randn(2, 1, 16000)

encoder = SincNet(pretrained=False).eval()
classifier = nn.Linear(2048, 2)
x = torch.randn(2, 1, 3200) 

z = encoder(x)
y = classifier(z)
print(z.shape, y.shape)