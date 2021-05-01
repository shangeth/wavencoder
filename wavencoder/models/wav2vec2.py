'''
Refer fairseq's wav2vec for more pretrained models
https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

Code from fairseq's repository is used in this file.
If you are using wavencoder only for the wav2vec models, then check the fairseq repository and can directly use their model/code.
'''
import torch
import torch.nn as nn
import fairseq
from tqdm import tqdm
import os
import urllib.request
from wavencoder.utils import _reporthook

class Wav2Vec2(nn.Module):
    def __init__(self, model_type='base', pretrained=True, pretrained_path=None, device=torch.device("cpu"), model_link=None):
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.model_links = {
            'base' : 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
            'large' : 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt',
            'xlsr53' : 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt'
        }
        self.model_link = model_link
        if not self.model_link:
            self.model_link = self.model_links[self.model_type]

        if pretrained:
            filename = self.model_link.split('/')[-1]
            if pretrained_path == None:
                if not os.path.exists(filename):
                    print(f'Downloading the pretrained weights from fairseq({self.model_link}) ...', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(self.model_link, filename, reporthook=_reporthook(t))
                model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([filename])
            else: 
                model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_path])
            self.model = model[0]
        else:
            print('Only Pretrained models possible, download the pretrained model and change the weights!!')

    def forward(self, x):
        x = x.squeeze(1)
        return self.model(x, features_only=True)