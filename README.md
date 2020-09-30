# WavEncoder

WavEncoder is a Python library for encoding raw audio with PyTorch backend.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install wavencoder.

```bash
pip install wavencoder

# currently its in test pypi
pip install -i https://test.pypi.org/simple/ wavencoder==0.1.2
```

## Usage

```python
import torch
import wavencoder

x = torch.randn(1, 16000) # [1, 16000]
encoder = wavencoder.models.Wav2Vec()
z = encoder(x) # [1, 512, 99]

classifier = wavencoder.models.MLPClassifier(infeature=512, out=2)
z_avg = torch.mean(z, 2) # [1, 512]
y_hat = classifier(z_avg) # [1, 2]
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
