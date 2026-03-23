import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files


class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int = 4, downsampling_factor: int = 320):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.downsampling_factor = downsampling_factor
        
        # Fetch tokens used by Torchaudio's pretrained decoder
        files = download_pretrained_files("librispeech-4-gram")
        with open(files.tokens, "r") as f:
            lines =[line.strip() for line in f]
            self.char_to_idx = {char: i for i, char in enumerate(lines)}

    def text_to_labels(self, text: str) -> torch.Tensor:
        # Standardize text to match torchaudio tokens (replace space with |)
        text = text.replace(" ", "|").upper()
        # Fallback to 1 (usually [UNK]) if char is missing
        return torch.tensor([self.char_to_idx.get(c, 1) for c in text], dtype=torch.long)

    def collate_fn(self, batch):
        audios = [item[0].squeeze(0) for item in batch]  # Shape: (T,)
        texts = [item[2] for item in batch]
        labels = [self.text_to_labels(t) for t in texts]
        
        # Pad sequences
        padded_audio = pad_sequence(audios, batch_first=True)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
        
        # Compute lengths
        audio_lengths = torch.tensor([len(a) for a in audios])
        label_lengths = torch.tensor([len(l) for l in labels])
        
        # Compute downsampled lengths for CTC (CRITICAL)
        input_lengths = audio_lengths // self.downsampling_factor
        
        # Create transformer padding mask (True where padded)
        max_len_downsampled = padded_audio.shape[1] // self.downsampling_factor
        padding_mask = torch.arange(max_len_downsampled).expand(len(batch), -1) >= input_lengths.unsqueeze(1)
        
        return (padded_audio, padded_labels, input_lengths, label_lengths, padding_mask)

    def test_collate_fn(self, batch):
        """Test loader passes raw text for WER computation"""
        padded_audio, padded_labels, input_lengths, label_lengths, padding_mask = self.collate_fn(batch)
        texts = [item[2].upper() for item in batch]
        return {
            "audio": padded_audio, 
            "text": texts, 
            "input_lengths": input_lengths,
            "padding_mask": padding_mask
        }

    def train_dataloader(self):
        ds = torchaudio.datasets.LIBRISPEECH(self.root_dir, url="train-clean-100", download=True)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=True)

    def test_dataloader(self):
        ds = torchaudio.datasets.LIBRISPEECH(self.root_dir, url="test-other", download=True) # Paper uses dev-other/test-other
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, collate_fn=self.test_collate_fn, num_workers=4)