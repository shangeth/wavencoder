import random
import torch
import torchaudio
from torchaudio.compliance import kaldi


class SpeedChange:
    def __init__(self, factor_range=(-0.15, 0.15), orig_freq=16000, p=0.5):
        self.factor_range = factor_range
        self.orig_freq = orig_freq
        self.p = p

    # @profile
    def __call__(self, wav):
        if random.random() < self.p:
            wav = wav.view(-1)
            warp_factor = random.uniform(self.factor_range[0], self.factor_range[1]) 
            samp_warp = self.orig_freq + self.orig_freq * warp_factor
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
        else:
            return wav