import torch.nn.functional as F
import torch
import os
import random
import torchaudio


class Reverberation:
    def __init__(self, ir_files_dir, max_reverb_len=24000, ir_rate=16000, p=0.5):
        self.ir_files_dir = ir_files_dir
        self.ir_files = os.listdir(self.ir_files_dir)
        self.max_reverb_len = max_reverb_len
        self.ir_rate = ir_rate
        self.p = p
    
    def load_IR(self):
        ir_file = random.choice(self.ir_files)
        ir_file = os.path.join(self.ir_files_dir, ir_file)
        IR, rate = torchaudio.load(ir_file)

        if IR.size(0) == 2:
            IR = IR[0]

        if rate != self.ir_rate:
            transformed = torchaudio.transforms.Resample(rate, self.ir_rate)(IR.view(1,-1))
            
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
        if random.random() < self.p:
            wav = wav.view(-1)
            IR, p_max = self.load_IR()
            Ex = torch.dot(wav, wav)
            rev = F.conv1d(wav.view(1, 1, -1), IR.view(1, 1, -1)).view(-1)
            Er = torch.dot(rev, rev)
            rev = self.shift(rev, -p_max)
            if Er>0:
                Eratio = torch.sqrt(Ex/Er)
            else:
                Eratio = 1.0
            rev = rev[:wav.shape[0]]
            rev = Eratio * rev
            return rev.view(1, -1)
        else:
            return wav


