import random
import torch
import torchaudio
import torchaudio.functional as F_audio
import pytorch_lightning as pl
import torch.distributed as dist
import numpy as np

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
        infinite=False,
    ):
        """
        Args:
            root_dir: Base directory for audio files.
            manifest_path: Path to manifest. Expected format per line: `<path> \t <samples> \t <text>`
            target_sample_rate: Desired audio sample rate.
            max_tokens: Maximum total samples in a batch (Wav2Vec2 limits this to prevent OOM).
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
        
        batches = []
        current_batch =[]
        current_max_samples = 0
        
        for item in buffer:
            # Wav2Vec2 determines batch fullness by checking if max_length * batch_size <= max_tokens
            next_max_samples = max(current_max_samples, item["samples"])
            next_batch_size = len(current_batch) + 1
            
            if next_max_samples * next_batch_size > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch =[item]
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
                # Yield 1D tensors for the feature extractor
                batch_data.append((waveform.squeeze(0), entry["text"]))
            
            yield batch_data


class CTCCollateFn:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor 

    def __call__(self, batch):
        audios, texts, raw_texts = [],[],[]

        for item in batch:
            waveform, text = item
            
            audios.append(waveform.numpy())
            
            tokens = torch.tensor(self.tokenizer(text), dtype=torch.long)
            texts.append(tokens)
            raw_texts.append(text)


        batch_features = self.feature_extractor(
            audios,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        padded_audios = batch_features["input_values"] # Shape: (B, T)
        
        # 1 means real audio, 0 means padding
        attention_mask = batch_features.get("attention_mask") 
        if attention_mask is None:
            attention_mask = torch.ones_like(padded_audios, dtype=torch.long)

        # 2. Pad text sequences with -100 so the CTC loss ignores them
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=-100) 

        return {
            "audio": padded_audios,              # Now properly padded and normalized!
            "labels": padded_texts,              # Shape: (B, L)
            "padding_mask": attention_mask,      # We map HF's attention mask to your Lightning model's expected `padding_mask` key
            "text": raw_texts                    # Raw string targets for WER calculation
        }


class LibriLightDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        feature_extractor,
        dev_other: str,
        dev_other_dir: str,
        dev_clean: str,
        dev_clean_dir: str,
        train: str,
        train_dir: str,
        test_clean: str = None,
        test_clean_dir: str = None,
        test_other: str = None,
        test_other_dir: str = None,
        max_tokens: int = 6400000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dev_other = dev_other
        self.dev_clean = dev_clean
        self.test_other = test_other
        self.test_clean = test_clean
        self.train = train

        self.dev_other_dir = dev_other_dir
        self.dev_clean_dir = dev_clean_dir
        self.test_other_dir = test_other_dir
        self.test_clean_dir = test_clean_dir
        self.train_dir = train_dir

        self.max_tokens = max_tokens
        self.num_workers = num_workers

        # Pass feature extractor directly to the Collator
        self.collate_fn = CTCCollateFn(tokenizer, feature_extractor)

    def train_dataloader(self):
        self.train_dataset = StreamingLibriLightDataset(
            root_dir=self.train_dir,
            manifest_path=self.train,
            max_tokens=self.max_tokens,
            infinite=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )

    def dev_clean_dataloader(self):
        self.dev_clean_dataset = StreamingLibriLightDataset(
            root_dir=self.dev_clean_dir,
            manifest_path=self.dev_clean,
            max_tokens=self.max_tokens,
            infinite=False,
        )

        return DataLoader(
            self.dev_clean_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )

    def dev_other_dataloader(self):
        self.dev_other_dataset = StreamingLibriLightDataset(
            root_dir=self.dev_other_dir,
            manifest_path=self.dev_other,
            max_tokens=self.max_tokens,
            infinite=False,
        )

        return DataLoader(
            self.dev_other_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )

    def test_clean_dataloader(self):
        self.test_clean_dataset = StreamingLibriLightDataset(
            root_dir=self.test_clean_dir,
            manifest_path=self.test_clean,
            max_tokens=self.max_tokens,
            infinite=False,
        )

        return DataLoader(
            self.test_clean_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )

    def test_other_dataloader(self):
        self.test_other_dataset = StreamingLibriLightDataset(
            root_dir=self.test_other_dir,
            manifest_path=self.test_other,
            max_tokens=self.max_tokens,
            infinite=False,
        )

        return DataLoader(
            self.test_other_dataset,
            batch_size=None,     
            shuffle=False,       
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )