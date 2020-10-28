__version__ = '0.0.6'



from wavencoder.models.wav2vec import Wav2Vec
from wavencoder.models.sincnet import SincNet
from wavencoder.models.lstm_classifier import LSTM_Classifier
from wavencoder.models.lstm_classifier import LSTM_Attn_Classifier
from wavencoder.models.baseline import CNN1d

from wavencoder.trainer.classification_trainer import train
from wavencoder.trainer.classification_trainer import test_predict_classifier
from wavencoder.trainer.classification_trainer import test_evaluate_classifier

from wavencoder.transforms.noise import AdditiveNoise
from wavencoder.transforms.speed import SpeedChange
from wavencoder.transforms.clip import Clipping
from wavencoder.transforms.compose import Compose
