import random
import torch
from torchaudio.compliance import kaldi

class SpeedChange:
    def __init__(self, factor_range=(-0.15, 0.15), orig_freq=16000):
        self.factor_range = factor_range
        self.orig_freq = orig_freq

    # @profile
    def __call__(self, wav):
        wav = wav.view(-1)
        warp_factor = random.random() * (self.factor_range[1] - self.factor_range[0]) + self.factor_range[0]
        samp_warp = wav.shape[0] + int(warp_factor * wav.shape[0])
        samp_warp = 10000
        rwav = kaldi.resample_waveform(wav.view(1, -1), self.orig_freq, samp_warp)
        rwav = rwav.view(-1)

        if len(rwav) > len(wav):
            mid_i = (len(rwav) // 2) - len(wav) // 2
            rwav = rwav[mid_i:mid_i + len(wav)]
        if len(rwav) < len(wav):
            diff = len(wav) - len(rwav)
            P = (len(wav) - len(rwav)) // 2
            if diff % 2 == 0:
                rwav = torch.cat((torch.zeros(P, ),
                                       rwav,
                                       torch.zeros(P, )),
                                      axis=0)
            else:
                rwav = torch.cat((torch.zeros(P, ),
                                       rwav,
                                       torch.zeros(P + 1, )),
                                      axis=0)
        rwav = rwav.view(1, -1)
        return rwav