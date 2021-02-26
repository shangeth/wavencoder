import numpy as np
import random

class TimeMask:
    '''
    shape of input (freq, time)
    '''
    def __init__(self, p=0.5, value=0.0, nmasks_range=(1, 5)):
        self.p = p
        self.value = value
        self.nmasks_range = nmasks_range

    def __call__(self, spectrogram):
        #[..., F, T]
        if random.random() < self.p:
            n_T = spectrogram.shape[2]
            nmasks = np.random.randint(self.nmasks_range[0], self.nmasks_range[1])
            for i in range(nmasks):
                t = int(np.random.uniform(low=0.0, high=n_T))
                spectrogram[:, :, t] = self.value
            return spectrogram
        else:
            return spectrogram


class FrequencyMask:
    def __init__(self, p=0.5, value=0.0, nmasks_range=(1, 5)):
        self.p = p
        self.value = value
        self.nmasks_range = nmasks_range

    def __call__(self, spectrogram):
        #[..., F, T]
        if random.random() < self.p:
            n_f = spectrogram.shape[1]
            nmasks = np.random.randint(self.nmasks_range[0], self.nmasks_range[1])
            for i in range(nmasks):
                f = int(np.random.uniform(low=0.0, high=n_f))
                spectrogram[:, f, :] = self.value
            return spectrogram
        else:
            return spectrogram

if __name__ == '__main__':
    pass
