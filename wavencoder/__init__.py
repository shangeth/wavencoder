__version__ = '0.0.6'



from wavencoder.models.wav2vec import Wav2Vec
# from wavencoder.models.wav2vec2 import Wav2Vec2Model
from wavencoder.models.sincnet import SincNet, SincConvLayer
from wavencoder.models.lstm_classifier import LSTM_Classifier
from wavencoder.models.lstm_classifier import LSTM_Attn_Classifier
from wavencoder.models.baseline import CNN1d
from wavencoder.models.rawnet import RawNet2Model

from wavencoder.layers.attention_layer import SoftAttention, DotAttention, AdditiveAttention, MultiplicativeAttention
from wavencoder.layers.sincnet_layer import SincConvLayer

from wavencoder.trainer.classification_trainer import train
from wavencoder.trainer.classification_trainer import test_predict_classifier
from wavencoder.trainer.classification_trainer import test_evaluate_classifier

from wavencoder.transforms.noise import AdditiveNoise
from wavencoder.transforms.speed import SpeedChange
from wavencoder.transforms.clip import Clipping
from wavencoder.transforms.pad_crop import Pad, Crop, PadCrop
from wavencoder.transforms.reverberation import Reverberation
from wavencoder.transforms.compose import Compose
from wavencoder.transforms.timeshift import TimeShift
from wavencoder.transforms.spec_augment import TimeMask, FrequencyMask

from wavencoder.utils import _reporthook
from wavencoder.utils import example_wav_file
from wavencoder.utils import download_noise_dataset
from wavencoder.utils import download_ir_dataset