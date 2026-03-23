import math
import random
import torch
import torchaudio
import torchaudio.functional as F_audio
import pytorch_lightning as pl
import torch.distributed as dist

from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class StreamingLibriLightDataset(IterableDataset):
    def __init__(
        self, 
        root_dir, 
        manifest_path, 
        target_sample_rate=16000, 
        max_tokens=6400000, 
        buffer_size=10000,
        infinite = False,
    ):
        """
        Args:
            root_dir: Base directory for audio files.
            manifest_path: Path to manifest. Expected format per line: `<path> \t <samples> \t <text>`
            target_sample_rate: Desired audio sample rate.
            max_tokens: Maximum total samples in a batch
            buffer_size: Number of files to buffer before bucketing and yielding.
        """
        self.root_dir = Path(root_dir)
        self.manifest_path = manifest_path
        self.target_sample_rate = target_sample_rate
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.infinite = infinite

    def _get_worker_and_node_info(self):
        # Handle Distributed Data Parallel (DDP) configuration
        if dist.is_available() and dist.is_initialized():
            node_rank = dist.get_rank()
            node_world_size = dist.get_world_size()
        else:
            node_rank = 0
            node_world_size = 1
            
        # Handle DataLoader multiple workers
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
            
        return node_rank, node_world_size, worker_id, num_workers

    def __iter__(self):
        node_rank, node_world_size, worker_id, num_workers = self._get_worker_and_node_info()
        
        total_workers = node_world_size * num_workers
        global_worker_id = node_rank * num_workers + worker_id
        
        buffer =[]
        while True:
            with open(self.manifest_path) as f:
                for i, line in enumerate(f):
                    if i % total_workers != global_worker_id:
                        continue
                        
                    parts = line.strip().split("\t") # Assumption: path, samples, text
                    if len(parts) >= 3:
                        audio_rel_path, num_samples_str, text = parts[0], parts[1], parts[2]
                        num_samples = int(num_samples_str)
                            
                        buffer.append({
                            "path": self.root_dir / audio_rel_path,
                            "samples": num_samples,
                            "text": text
                        })
                        
                    if len(buffer) == self.buffer_size:
                        yield from self._process_buffer(buffer)
                        buffer =[]
                        
                if len(buffer) > 0:
                    yield from self._process_buffer(buffer)
                if not self.infinite:
                    break

    def _process_buffer(self, buffer):
        buffer.sort(key=lambda x: x["samples"])
        
        batches =[]
        current_batch =[]
        current_max_samples = 0
        
        for item in buffer:
            # Wav2Vec2 determines batch fullness by checking if max_length * batch_size <= max_tokens
            next_max_samples = max(current_max_samples, item["samples"])
            next_batch_size = len(current_batch) + 1
            
            if next_max_samples * next_batch_size > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_max_samples = item["samples"]
            else:
                current_batch.append(item)
                current_max_samples = next_max_samples
                
        if current_batch:
            batches.append(current_batch)
            
        random.shuffle(batches)
        
        for batch in batches:
            batch_data =[]
            for entry in batch:
                waveform, sample_rate = torchaudio.load(entry["path"])
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sample_rate != self.target_sample_rate:
                    waveform = F_audio.resample(waveform, orig_freq=sample_rate, new_freq=self.target_sample_rate)
                batch_data.append((waveform.squeeze(0), entry["text"]))
            
            yield batch_data


class CTCCollateFn:
    def __init__(self, tokenizer, audio_token_func):
        self.tokenizer = tokenizer
        self.audio_token_func = audio_token_func 

    def __call__(self, batch):
        audios, texts, raw_texts = [], [], []
        input_lengths, label_lengths = [],[]

        for item in batch:
            waveform, text = item
            
            audios.append(waveform)
            input_lengths.append(waveform.shape[-1])
            
            tokens = torch.tensor(self.tokenizer(text), dtype=torch.long)
            texts.append(tokens)
            label_lengths.append(tokens.shape[-1])
            raw_texts.append(text)

        padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=-100) 

        batch_size, max_len = padded_audios.size()
        padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= torch.tensor(input_lengths).unsqueeze(1)
        
        downsampled_lengths = torch.tensor([self.audio_token_func(l) for l in input_lengths])
        max_downsampled_len = self.audio_token_func(max_len)
        attention_mask_tokens = torch.arange(max_downsampled_len).expand(batch_size, max_downsampled_len) >= downsampled_lengths.unsqueeze(1)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return {
            "audio": padded_audios.unsqueeze(1),
            "labels": padded_texts,
            "input_lengths": input_lengths,
            "label_lengths": label_lengths,
            "padding_mask": ~padding_mask,
            "attention_mask": attention_mask_tokens,
            "text": raw_texts
        }


class LibriLightDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        val_data_root: str,
        test_data_root : str,
        train_manifest_path: str,
        val_manifest_path:str,
        test_manifest_path:str,
        tokenizer,
        audio_token_func,
        max_tokens: int = 6400000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_manifest_path = train_manifest_path
        self.val_manifest_path = val_manifest_path
        self.test_manifest_path = test_manifest_path
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.collate_fn = CTCCollateFn(tokenizer, audio_token_func)
        self.val_data_root = val_data_root
        self.test_data_root = test_data_root

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = StreamingLibriLightDataset(
                root_dir=self.data_root, 
                manifest_path=self.train_manifest_path,
                max_tokens=self.max_tokens,
                infinite=True
            )
            
            self.val_dataset = StreamingLibriLightDataset(
                root_dir=self.val_data_root, 
                manifest_path=self.val_manifest_path,
                max_tokens=self.max_tokens,
                infinite=False
            )
        if stage == "test":
            self.test_dataset = StreamingLibriLightDataset(
                root_dir=self.test_data_root, 
                manifest_path=self.test_manifest_path,
                max_tokens=self.max_tokens,
                infinite=False
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )