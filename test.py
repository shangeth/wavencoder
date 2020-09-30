from wavencoder.models import Wav2Vec
# from wavencoder.models import SincNet
from wavencoder.models import LSTM_Classifier, LSTM_Attn_Classifier
import torch

encoder = Wav2Vec(pretrained=False)
# encoder = SincNet(pretrained=True).eval()
classifier = LSTM_Attn_Classifier(512, 64, 2)
x = torch.randn(2, 16000) 
z = encoder(x)
y, attn_weights = classifier(z)
print(z.shape, y.shape, attn_weights.shape)