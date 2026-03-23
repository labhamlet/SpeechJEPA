import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files



def get_tri_state_schedule(optimizer, total_steps, phase_ratio=(0.1, 0.4, 0.5)):
    """Matches the exact Wav2Vec 2.0 paper schedule: 10% warmup, 40% hold, 50% decay"""
    warmup_steps = int(phase_ratio[0] * total_steps)
    hold_steps = int(phase_ratio[1] * total_steps)
    decay_steps = total_steps - warmup_steps - hold_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < (warmup_steps + hold_steps):
            return 1.0
        else:
            step_in_decay = current_step - (warmup_steps + hold_steps)
            return max(0.0, float(decay_steps - step_in_decay) / float(max(1, decay_steps)))

    return LambdaLR(optimizer, lr_lambda)