import torchaudio
import os
import random
import torch
import torch.nn.functional as F

class AdditiveNoise:
    def __init__(self, noise_dir, snr_levels=[5, 10]):
        self.noise_dir = noise_dir
        self.snr_levels = snr_levels

    def compute_snr_k(self, wav, noise, snr):
        Ex = torch.dot(wav.view(-1), wav.view(-1))
        En = torch.dot(noise.view(-1), noise.view(-1))
        if En>0:
            K = torch.sqrt(Ex/(En*10**(snr/10.)))
        else:
            K = 1.0
        return K, Ex, En

    def norm_energy(self, osignal, ienergy, eps=1e-14):
        oenergy = torch.dot(osignal.view(-1), osignal.view(-1))
        return torch.sqrt(ienergy / (oenergy + eps)) * osignal

    def __call__(self, wav):
        noise_file = random.choice(os.listdir(self.noise_dir))
        noise, _ = torchaudio.load(self.noise_dir + '/' + noise_file)
        len_wav = wav.shape[1]
        len_noise = noise.shape[1]
        delta = len_noise - len_wav

        if delta > 0:
            i = random.randint(0, len_noise-len_wav)
            noise = noise[:, i:i+len_wav]
        elif delta < 0:
            noise = F.pad(noise, (int(-delta/2), int(-delta) - int(-delta/2)), 'constant', 0)

        snr = random.choice(self.snr_levels)
        K, Ex, En = self.compute_snr_k(wav, noise, snr)
        scaled_noise = K * noise

        if En > 0:
            noisy = wav + scaled_noise
            noisy = self.norm_energy(noisy, Ex)
        else:
            noisy = wav

        return noisy

    
if __name__ == "__main__":
    add = AdditiveNoise('/home/shangeth/Downloads/Dataset/DLIVING_16k/DLIVING')
    audio = torchaudio.load('/home/shangeth/Downloads/Dataset/TIMIT/Wav_Data/TRAIN/FAEM0_SA2.WAV')
    noisy = add(audio)
