![PyPI](https://img.shields.io/pypi/v/wavencoder)
![PyPI - Downloads](https://img.shields.io/pypi/dw/wavencoder?logo=PyPi&style=plastic)
![visitors](https://visitor-badge.glitch.me/badge?page_id=page.id)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/shangeth/wavencoder/issues)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wavencoder)
![GitHub last commit](https://img.shields.io/github/last-commit/shangeth/wavencoder)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/shangeth/wavencoder)
![GitHub](https://img.shields.io/github/license/shangeth/wavencoder)
[![Gitter](https://badges.gitter.im/wavencoder/community.svg)](https://gitter.im/wavencoder/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Twitter Follow](https://img.shields.io/twitter/follow/shangethr?style=social)


# WavEncoder

WavEncoder is a Python library for encoding audio signal, transforms for audio augmention and training audio classification models with PyTorch backend.

## Package Contents

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Layers</th>
    <th class="tg-7btt">Models</th>
    <th class="tg-7btt">Transforms</th>
    <th class="tg-7btt">Trainer and utils</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">
        <ul>
            <li>Attention</li>
            <ul>
                <li>Dot</li>
                <li>Soft</li>
                <li>Additive</li>
                <li>Multiplicative</li>
            </ul>
            <li>SincNet layer</li>
            <li>Time Delay Neural Network(TDNN)</li>
        </ul>
    </td>
    <td class="tg-0pky">
        <ul>
            <li>PreTrained</li>
                <ul>
                    <li>wav2vec</li>
                    <li>SincNet</li>
                    <li>RawNet</li>
                </ul>
            <li>Baseline</li>
                <ul>
                    <li>1DCNN</li>
                    <li>LSTM Classifier</li>
                    <li>LSTM Attention Classifier</li>
                </ul>
        </ul>
    </td>
    <td class="tg-0pky">
        <ul>
            <li>Noise(Environmet/Gaussian White Noise)</li>
            <li>Speed Change</li>
            <li>PadCrop</li>
            <li>Clip</li>
            <li>Reverberation</li>
            <li>TimeShift</li>
            <li>TimeMask</li>
            <li>FrequencyMask</li>
        </ul>
    </td>
    <td class="tg-0pky">
        <ul>
            <li>Classification Trainer</li>
            <li>Classification Testing</li>
            <li>Download Noise Dataset</li>
            <li>Download Impulse Response Dataset</li>
        </ul>
    </td>
  </tr>
</tbody>
</table>



## Wav Models to be added
- [x] wav2vec [[1]](#1)
- [ ] wav2vec2 [[2]](#2)
- [x] SincNet [[3]](#3)
- [ ] PASE [[4]](#4)
- [ ] MockingJay [[5]](#5)
- [x] RawNet [[6]](#6)
- [ ] GaborNet [[7]](#7)
- [ ] LEAF [[8]](#8)
- [x] CNN-1D
- [x] CNN-LSTM
- [x] CNN-LSTM-Attn
- [ ] CNN-Transformer

Check the [Demo Colab Notebook](https://colab.research.google.com/drive/1Jv9cH4H0xB2To1rihFz-Z-JaK-6ilq12?usp=sharing).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wavencoder.

```bash
pip install wavencoder
```

## Usage
### Import pretrained encoder, baseline models and classifiers
```python
import torch
import wavencoder

x = torch.randn(1, 16000) # [1, 16000]
encoder = wavencoder.models.Wav2Vec(pretrained=True)
z = encoder(x) # [1, 512, 98]

classifier = wavencoder.models.LSTM_Attn_Classifier(512, 64, 2,                          
                                                    return_attn_weights=True, 
                                                    attn_type='soft')
y_hat, attn_weights = classifier(z) # [1, 2], [1, 98]

```

### Use wavencoder with PyTorch Sequential or class modules
```python
import torch
import torch.nn as nn
import wavencoder

model = nn.Sequential(
        wavencoder.models.Wav2Vec(),
        wavencoder.models.LSTM_Attn_Classifier(512, 64, 2,                          
                                               return_attn_weights=True, 
                                               attn_type='soft')
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

### Add Transforms to your DataLoader for Augmentation/Processing the wav signal
```python
from wavencoder.transforms import Compose, AdditiveNoise, SpeedChange, Clipping, PadCrop, Reverberation

audio, _ = torchaudio.load('test.wav')

transforms = Compose([
                    AdditiveNoise('path-to-noise-folder', p=0.5, snr_levels=[5, 10, 15], p=0.5), 
                    SpeedChange(factor_range=(-0.5, 0.0), p=0.5), 
                    Clipping(p=0.5),
                    PadCrop(48000, crop_position='random', pad_position='random') 
                    ])

transformed_audio = transforms(audio)

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


