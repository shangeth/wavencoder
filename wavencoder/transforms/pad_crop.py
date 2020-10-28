import torch.nn.functional as F
import random

class Pad:
    def __init__(self, pad_length=16000, mode='constant', value=0, pad_position='center'):
        self.pad_length = pad_length
        self.mode = mode
        self.value = value
        self.pad_position = pad_position

    def __call__(self, wav):  
        wav_length = wav.shape[1]
        delta = self.pad_length - wav_length
        if self.pad_position == 'center':
            left_crop_len = int(delta/2)
            right_crop_len = delta - int(delta/2)
        elif self.pad_position == 'right':
            left_crop_len = 0
            right_crop_len = delta
        elif self.pad_position == 'left':
            left_crop_len = delta
            right_crop_len = 0
        elif self.pad_position == 'random':
            left_crop_len = int(random.random() * delta)
            right_crop_len = delta - left_crop_len

        wav = F.pad(wav, (left_crop_len, right_crop_len), self.mode, self.value)
        return wav

class Crop:
    def __init__(self, crop_length=16000, crop_position='center'):
        self.crop_length = crop_length
        self.crop_position = crop_position

    def __call__(self, wav):  
        wav_length = wav.shape[1]
        delta = wav_length - self.crop_length

        if self.crop_position == 'left':
            i = 0
        
        elif self.crop_position == 'left':
            i = delta
        
        elif self.crop_position == 'center':
            i = int(delta/2)
        
        elif self.crop_position == 'random':
            i = random.randint(0, delta)
        
        wav = wav[:, i:i+self.crop_length]
        return wav

class PadCrop:
    def __init__(self, pad_crop_length=16000, crop_position='center',
        pad_mode='constant', pad_value=0, pad_position='center'):
        
        self.pad_crop_length = pad_crop_length
        self.crop_position = crop_position
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.pad_position = pad_position
        self.crop_transform = Crop(crop_length=self.pad_crop_length, crop_position=self.crop_position)
        self.pad_transform = Pad(pad_length=self.pad_crop_length, mode=self.pad_mode, value=self.pad_value, pad_position=self.pad_position)

    def __call__(self, wav):  
        wav_length = wav.shape[1]
        delta = wav_length - self.pad_crop_length

        if delta>0:
            wav = self.crop_transform(wav)
        elif delta<0:
            wav = self.pad_transform(wav)
        return wav

    