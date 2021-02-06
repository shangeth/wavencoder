import torch.nn.functional as F
import torch
import os
import random
import torchaudio
import scipy.io as io

class Reverberation:
    def __init__(self, ir_files_dir, max_reverb_len=24000, mat_dict_key=None):
        self.ir_files_dir = ir_files_dir
        self.ir_files = os.listdir(self.ir_files_dir)
        self.max_reverb_len = max_reverb_len
        self.mat_dict_key = mat_dict_key
    
    def load_IR(self):
        ir_file = random.choice(self.ir_files)
        ir_file = os.path.join(self.ir_files_dir, ir_file)
        if ir_file.endswith('.mat'):
            data= io.loadmat(ir_file)
            IR = torch.from_numpy(data[self.mat_dict_key]).view(-1).float()
        else:
            IR, _ = torchaudio.load(ir_file)
            IR = IR.view(-1)
        IR = IR[:self.max_reverb_len]

        if torch.max(IR)>0:
            IR = IR/torch.abs(torch.max(IR))
        p_max = torch.argmax(torch.abs(IR))
        return IR, p_max

    def shift(self, xs, n):
        e = torch.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def __call__(self, wav):
        wav = wav.view(-1)
        IR, p_max = self.load_IR()
        Ex = torch.dot(wav, wav)
        rev = F.conv1d(wav.view(1, 1, -1), IR.view(1, 1, -1)).view(-1)
        Er = torch.dot(rev, rev)
        # rev = self.shift(rev, -p_max)
        if Er>0:
            Eratio = torch.sqrt(Ex/Er)
        else:
            Eratio = 1.0
        rev = rev[:wav.shape[0]]
        rev = Eratio * rev
        return rev.view(1, -1)


