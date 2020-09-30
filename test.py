from wavencoder.models import Wav2Vec
from wavencoder.models import SincNet
import torch

# encoder = Wav2Vec(pretrained=True)
encoder = SincNet(pretrained=True).eval()
x = torch.randn(2, 3200) 
z = encoder(x)
print(z.shape)