![PyPI - Downloads](https://img.shields.io/pypi/dw/wavencoder?logo=PyPi&style=plastic)
![visitors](https://visitor-badge.glitch.me/badge?page_id=page.id)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wavencoder)
![GitHub last commit](https://img.shields.io/github/last-commit/shangeth/wavencoder)
![GitHub top language](https://img.shields.io/github/languages/top/shangeth/wavencoder)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/shangeth/wavencoder)
![Twitter Follow](https://img.shields.io/twitter/follow/shangethr?style=social)
[![HitCount](http://hits.dwyl.com/shangeth/wavencoder.svg)](http://hits.dwyl.com/shangeth/wavencoder)
![Gitter](https://img.shields.io/gitter/room/shangeth/wavencoder)


# WavEncoder

WavEncoder is a Python library for encoding raw audio with PyTorch backend.

## Wav Models to be added
- [x] wav2vec [[1]](#1)
- [ ] wav2vec2 [[2]](#2)
- [x] SincNet [[3]](#3)
- [ ] PASE [[4]](#4)
- [ ] MockingJay [[5]](#5)
- [ ] RawNet [[6]](#6)
- [x] CNN-1D
- [x] CNN-LSTM
- [x] CNN-LSTM-Attn
- [ ] CNN-Transformer

Check the [Demo Colab Notebook](https://colab.research.google.com/drive/1Jv9cH4H0xB2To1rihFz-Z-JaK-6ilq12?usp=sharing).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wavencoder.

```bash
pip install fairseq
pip install wavencoder
```

## Usage
### Import pretrained encoder models and classifiers
```python
import torch
import wavencoder

x = torch.randn(1, 16000) # [1, 16000]
encoder = wavencoder.models.Wav2Vec(pretrained=True)
z = encoder(x) # [1, 512, 98]

classifier = wavencoder.models.LSTM_Attn_Classifier(512, 64, 2)
y_hat, attn_weights = classifier(z) # [1, 2], [1, 98]

```

### Use wavencoder with PyTorch Sequential or class modules
```python
import torch
import torch.nn as nn
import wavencoder

model = nn.Sequential(
        wavencoder.models.Wav2Vec(),
        wavencoder.models.LSTM_Attn_Classifier(512, 64, 2)
)

x = torch.randn(1, 16000) # [1, 16000]
y_hat, attn_weights = model(x) # [1, 2], [1, 98]
```

```python
import torch
import torch.nn as nn
import wavencoder

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.mean(z, dim=2)
        out = self.classifier(z)
        return out

model = AudioClassifier()
x = torch.randn(1, 16000) # [1, 16000]
y_hat = model(x) # [1, 2]
```
### Train the encoder-classifier models
```python
from wavencoder.models import Wav2Vec, LSTM_Attn_Classifier
from wavencoder.trainer import train, test_evaluate_classifier, test_predict_classifier

model = nn.Sequential(
    Wav2Vec(pretrained=False),
    LSTM_Attn_Classifier(512, 64, 2)
)

trainloader = ...
valloader = ...
testloader = ...

trained_model, train_dict = train(model, trainloader, valloader, n_epochs=20)
test_prediction_dict = test_predict_classifier(trained_model, testloader)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](LICENSE)


## Reference
|     | Paper                                                                                                                                                    | Code                                                                                                 |
|-----|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [1] | [Wav2Vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)                                                            | [GitHub](https://github.com/pytorch/fairseq)                                                         |
| [2] | [Wav2vec 2.0: Learning the structure of speech from raw audio](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) | [GitHub](https://github.com/pytorch/fairseq)                                                         |
| [3] | [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)                                                                   | [GitHub](https://github.com/mravanelli/SincNet)                                                      |
| [4] | [Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks](https://arxiv.org/abs/1904.03416)                                 | [GitHub](https://github.com/santi-pdp/pase)                                                          |
| [5] | [Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders](https://arxiv.org/abs/1910.12638)                 | [GitHub](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning ) |
| [6] | [Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms](https://arxiv.org/abs/2004.00526)               | [GitHub](https://github.com/Jungjee/RawNet)                                                          |
