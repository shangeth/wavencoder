from wavencoder.models import Wav2Vec
import torch

encoder = Wav2Vec(pretrained=True, pretrained_path='/home/shangeth/Documents/GitHub/wav2vec_large.pt')
x = torch.randn(1, 16000) 
z = encoder(x)
print(z.shape)