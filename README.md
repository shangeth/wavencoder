![Twitter Follow](https://img.shields.io/twitter/follow/shangethr?style=social)

# WavEncoder

WavEncoder is a Python library for encoding raw audio with PyTorch backend.

## Models to be added
- [x] wav2vec [[1]](#1)
- [ ] wav2vec2 [[2]](#2)
- [x] SincNet [[3]](#3)
- [ ] PASE [[4]](#4)
- [ ] MockingJay [[5]](#5)

Check the [Demo Colab Notebook](https://colab.research.google.com/drive/1Jv9cH4H0xB2To1rihFz-Z-JaK-6ilq12?usp=sharing).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wavencoder.

```bash
pip install fairseq
pip install wavencoder
```

## Usage

```python
import torch
import wavencoder

x = torch.randn(1, 16000) # [1, 16000]
encoder = wavencoder.models.Wav2Vec()
z = encoder(x) # [1, 512, 98]

classifier = wavencoder.models.LSTM_Attn_Classifier(512, 64, 2)
y_hat, attn_weights = classifier(z) # [1, 2], [1, 98]

```

```python
import torch
import torch.nn as nn
import wavencoder

model = nn.Sequential(
        wavencoder.models.Wav2Vec(),
        wavencoder.models.LSTM_Attn_Classifier(512, 64, 2)
)

x = torch.randn(1, 16000)
y_hat, attn_weights = model(x)

print(y_hat.shape, attn_weights.shape)
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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](LICENSE)


## Reference
|                   | Paper                                                                                                                                                    | Code                                                                                                 |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| <a id="1">[1]</a> | [Wav2Vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)                                                            | [GitHub](https://github.com/pytorch/fairseq)                                                         |
| <a id="2">[2]</a> | [Wav2vec 2.0: Learning the structure of speech from raw audio](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) | [GitHub](https://github.com/pytorch/fairseq)                                                         |
| <a id="3">[3]</a> | [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)                                                                   | [GitHub](https://github.com/mravanelli/SincNet)                                                      |
| <a id="4">[4]</a> | [Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks](https://arxiv.org/abs/1904.03416)                                 | [GitHub](https://github.com/santi-pdp/pase)                                                          |
| <a id="5">[5]</a> | [Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders](https://arxiv.org/abs/1910.12638)                 | [GitHub](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning ) |
