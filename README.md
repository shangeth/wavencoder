# WavEncoder

WavEncoder is a Python library for encoding raw audio with PyTorch backend.

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

classifier = wavencoder.models.MLPClassifier(infeature=512, out=2)
z_avg = torch.mean(z, 2) # [1, 512]
y_hat = classifier(z_avg) # [1, 2]
```

```python
import torch
import torch.nn as nn
import wavencoder

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.mean(z, dim=2)
        out = self.classifier(z)
        return out

model = AudioClassifier()
x = torch.randn(1, 16000)
y_hat = model(x)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)