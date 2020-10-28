import random
import torch

class Clipping(object):

    def __init__(self, clip_factors=[0.3, 0.4, 0.5],
                 report=False):
        self.clip_factors = clip_factors
        self.report = report

    #@profile
    def __call__(self, wav):
        cf = random.choice(self.clip_factors)
        clipped_wav = torch.clamp(wav.view(-1),cf*torch.min(wav),cf*torch.max(wav))
        return clipped_wav

    def __repr__(self):
        attrs = '(clip_factors={})'.format(
            self.clip_factors
        )
        return self.__class__.__name__ + attrs