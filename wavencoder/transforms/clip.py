import random
import torch

class Clipping(object):

    def __init__(self, clip_factors=[0.3, 0.4, 0.5], p=0.5):
        self.clip_factors = clip_factors
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            cf = random.choice(self.clip_factors)
            clipped_wav = torch.clamp(wav.view(-1),cf*torch.min(wav),cf*torch.max(wav))
            clipped_wav =clipped_wav.view(1, -1)
            return clipped_wav
        else:
            return wav


    def __repr__(self):
        attrs = '(clip_factors={})'.format(
            self.clip_factors
        )
        return self.__class__.__name__ + attrs