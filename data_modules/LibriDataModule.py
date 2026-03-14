import webdataset as wds
import speechbrain as sb 
import pytorch_lightning as pl 
import torch 

from torch.utils.data import DataLoader

from functools import partial 

from .dataset_functions import normalize_audio

from .utils import _get_feat_extract_output_lengths, visualize_masks


import webdataset as wds
import speechbrain as sb 
import pytorch_lightning as pl 
import torch 

from torch.utils.data import DataLoader

from functools import partial 

from .dataset_functions import normalize_audio

from .utils import _get_feat_extract_output_lengths, visualize_masks

import contextlib

class SSLDataModule(pl.LightningDataModule):
    BUCKET_BOUNDARIES = [32000,
                        48000, 
                        64000, 
                        80000,
                        96000, 
                        112000,
                        128000, 
                        144000,
                        160000, 
                        176000,
                        192000, 
                        208000,
                        224000, 
                        250000]
    def __init__(
        self,
        masker,
        data_dir: str,
        min_sample_len: int = 32000,
        max_sample_len: int = 250000,
        target_batch_size: int = 1_400_000,
        max_batch_size : int = 1_500_000,
        loudness_normalize: bool = True,
        conv_kernel: list =[10, 3, 3, 3, 3, 2, 2],
        conv_stride: list =[5, 2, 2, 2, 2, 2, 2],
        target_masks_per_context: int = 4,
        bucket_limits: bool = False,
        pin_memory: bool = True,
        num_workers : int = 16
    ):
        super().__init__()
        #With compilation, make sure that we compile not too much!
        self.save_hyperparameters(ignore=['masker'])
        self.masker = masker
        cfg_dict = {
            "conv_kernel": self.hparams.conv_kernel, 
            "conv_stride": self.hparams.conv_stride
        }
        self.token_func = partial(_get_feat_extract_output_lengths, cfg=cfg_dict)
        self.bucket_limits = bucket_limits
        if self.bucket_limits:
            print("Bucket Limits are turned on")
    def _crop_audio(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.ndim == 1
        if audio.shape[0] > self.hparams.max_sample_len:
            max_start = audio.shape[0] - self.hparams.max_sample_len
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            audio = audio[start_idx : start_idx + self.hparams.max_sample_len]
        return audio
    
    def _augment_sample(self, sample):
        audio, _ = sample["signal"]
        if audio.ndim == 2:
            audio = audio[0] # Convert [N, T] to [T] by taking the first mic.

        if self.hparams.loudness_normalize:
            audio = normalize_audio(audio)
        
        audio = self._crop_audio(audio)
        nr_tokens = self.token_func(audio.shape[-1]).item()
        
        # masker usually expects [B, T, C], we simulate B=1
        while True:
            try:
                ctx_mask, tgt_mask, ctx_tgt_masks = self.masker(
                batch_size=1, n_times=nr_tokens, in_channels=1
                )
                break
            except:  # noqa: E722
                pass

        
        return {
            "signal": audio,
            "ctx_mask": ctx_mask.squeeze(0), 
            "tgt_mask": tgt_mask.squeeze(0), 
            "ctx_tgt_masks": ctx_tgt_masks.squeeze(0)
        }

    @classmethod
    def get_bucket_length(cls, length: int) -> int:
        for b in cls.BUCKET_BOUNDARIES:
            if length <= b:
                return b
        return cls.BUCKET_BOUNDARIES[-1]

    @staticmethod
    def custom_collate_fn(batch, token_func, target_masks_per_ctx, bucket_limits):
        """
        Groups dictionary samples into a single batch with padding.
        """
        batch_size = len(batch)
        # Access using dictionary keys
        lengths = [item["signal"].shape[-1] for item in batch]
        max_len = max(lengths)
        if bucket_limits:
            max_len = SSLDataModule.get_bucket_length(max_len)
        padded_audio = torch.zeros(batch_size, max_len)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        # Calculate max tokens based on the longest audio in this batch
        nr_of_tokens_per_padded_audio = token_func(max_len).item()
        teacher_padding_mask = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        # Initialize masks: ctx/ctx_tgt as True (masked), tgt as False
        ctx_masks = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        ctx_tgt_masks = torch.ones(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        tgt_masks = torch.zeros(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        total_batch_len = 0

        for i, item in enumerate(batch):        
            sig = item["signal"]
            length = lengths[i]
            total_batch_len += length
            #The padding mask is opposite of transformer encoders
            #Padding mask True means that it is not masked.
            padded_audio[i, :length] = sig
            padding_mask[i, :length] = True 

            #How many tokens did this audio have before padding.   
            nr_of_tokens = item["ctx_mask"].shape[-1]
            
            ctx_masks[i, :nr_of_tokens] = item["ctx_mask"]
            ctx_tgt_masks[i, :, :nr_of_tokens] = item["ctx_tgt_masks"]
            tgt_masks[i, :, :nr_of_tokens] = item["tgt_mask"]
            #Okay to attend to the real audio!
            teacher_padding_mask[i, :nr_of_tokens] = False

        return {
            "audio": padded_audio,
            "padding_mask": padding_mask,
            "teacher_padding_mask" : teacher_padding_mask,
            "ctx_mask": ctx_masks,
            "tgt_mask": tgt_masks,
            "ctx_tgt_mask": ctx_tgt_masks
        }

    def make_web_dataset(self, path: str):
        bound_collate = partial(
            self.custom_collate_fn, 
            token_func=self.token_func, 
            target_masks_per_ctx=self.hparams.target_masks_per_context,
            bucket_limits=self.bucket_limits
        )

        dataset = (
            wds.WebDataset(path, resampled=True, handler=wds.warn_and_continue)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .rename(signal="flac")
            .select(lambda x: x["signal"][0].shape[-1] >= self.hparams.min_sample_len)
            .map(self._augment_sample)
            .repeat()
            .compose(
                partial(
                    sb.dataio.iterators.dynamic_bucketed_batch,
                    len_key="signal",
                    buffersize=8192,
                    collate_fn=bound_collate,
                    sampler_kwargs={
                        "target_batch_numel": self.hparams.target_batch_size,
                        "max_batch_numel": self.hparams.max_batch_size,
                    },
                )
            )
        )

        return dataset
    
    def setup(self, stage: str = None):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                self.hparams.data_dir
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=False,
            num_workers=self.hparams.num_workers,
            prefetch_factor=2
        )
        return loader
    