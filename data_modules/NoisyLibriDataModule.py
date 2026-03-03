import webdataset as wds
import speechbrain as sb 
import pytorch_lightning as pl 
import torch 

from typing import Optional

from torch.utils.data import DataLoader

from functools import partial 

from .utils import _get_feat_extract_output_lengths
from .dataset_functions import normalize_audio
from .scene_module.generate_scenes import fade_noise
import torchaudio

from .managers import RIRManager, NoiseManager

class NoisySSLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        masker,
        data_dir: str,
        rir_path : str,
        noise_path : str, 
        min_sample_len: int = 32000,
        max_sample_len: int = 250000,
        target_batch_size: int = 1_400_000,
        max_batch_size : int = 1_500_000,
        data_sr : int = 16000,
        noise_and_rir_sr : int = 32000,
        snr_low : float = -5.0, 
        snr_high: float = 20.0,
        max_tokens: int = 1_500_000,
        loudness_normalize: bool = True,
        conv_kernel: list =[10, 3, 3, 3, 3, 2, 2],
        conv_stride: list =[5, 2, 2, 2, 2, 2, 2],
        target_masks_per_context: int = 4,
        pin_memory: bool = True,
        num_workers : int = 16
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['masker'])
        self.masker = masker
        cfg_dict = {
            "conv_kernel": self.hparams.conv_kernel, 
            "conv_stride": self.hparams.conv_stride
        }
        self.token_func = partial(_get_feat_extract_output_lengths, cfg=cfg_dict)
        self.resampling_factor = noise_and_rir_sr / data_sr
        self.resample_sr = noise_and_rir_sr
        self.resampler = self._build_resampler(data_sr, noise_and_rir_sr)
        self.noise_loader = NoiseManager(noise_path).start()
        self.rir_loader = RIRManager(rir_path).start()

    def _build_resampler(self, sr: int, resample_sr: int) -> torchaudio.transforms.Resample:
        return torchaudio.transforms.Resample(
            sr,
            resample_sr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            dtype=torch.float32,
            beta=14.769656459379492,
        )

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

        audio = self._crop_audio(audio)
        audio = self.resampler(audio)
        nr_tokens = self.token_func(audio.shape[-1]).item()
        
        rirs = next(self.rir_loader)
        noise, noise_start_idx, noise_length = self._prepare_noise(audio)
        snr = torch.distributions.uniform.Uniform(self.hparams.snr_low, self.hparams.snr_high).sample().item()
        source_rir = rirs[0] 
        noise_rirs = rirs[1:]
        nr_tokens = self.token_func(int(audio.shape[-1] / self.resampling_factor)).item()

        # masker usually expects [B, T, C], we simulate B=1
        ctx_mask, tgt_mask, ctx_tgt_masks = self.masker(
            batch_size=1, n_times=nr_tokens, in_channels=1
        )
        
        return {
            "signal": audio,
            "noise" : noise, 
            "noise_start_idx" : noise_start_idx,
            "noise_length" : noise_length,
            "snr": snr,
            "source_rir": source_rir,
            "noise_rirs" : noise_rirs,
            "ctx_mask": ctx_mask.squeeze(0), 
            "tgt_mask": tgt_mask.squeeze(0), 
            "ctx_tgt_masks": ctx_tgt_masks.squeeze(0)
        }


    def _prepare_noise(
        self, audio: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], int, int, Optional[float]]:
        
        noise = next(self.noise_loader)
        noise = normalize_audio(noise)
        noise = fade_noise(noise, audio, self.resample_sr)
        noise_length = noise.shape[-1]
        noise_start_idx = 0

        if audio.shape[-1] > noise.shape[-1]:
            noise_start_idx = torch.randint(
                0, audio.shape[-1] - noise.shape[-1], (1,)
            ).item()
            padded_noise = torch.zeros_like(audio)
            padded_noise[noise_start_idx : noise_start_idx + noise.shape[-1]] = noise
            noise = padded_noise

        return noise, noise_start_idx, noise_length

    @staticmethod
    def custom_collate_fn(batch, token_func, target_masks_per_ctx, downsampling_rate):
        """
        Groups dictionary samples into a single batch with padding.
        """
        batch_size = len(batch)
        lengths = [item["signal"].shape[-1] for item in batch]
        max_len = max(lengths)

        padded_audio = torch.zeros(batch_size, max_len)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        padded_noise = torch.zeros(batch_size, max_len)

        nr_of_tokens_per_padded_audio = token_func(int(max_len / downsampling_rate)).item()

        #Initialzie everything as being masked.
        ctx_masks = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        teacher_padding_mask = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        ctx_tgt_masks = torch.ones(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)
        #False means that we do not propagate loss through these target indices.
        tgt_masks = torch.zeros(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype= torch.bool)
        source_rirs, noise_lengths, noise_start_idxs, noise_rirs_list, snrs = [], [], [], [],[]


        for i, item in enumerate(batch):        
            nr_of_tokens = item["ctx_mask"].shape[-1]
            assert nr_of_tokens_per_padded_audio >= nr_of_tokens, "Something went wrong with the padding"

            length = lengths[i]

            padded_audio[i, :length] = item["signal"]
            padding_mask[i, :length] = True # Mark actual audio frames as True, this is from huggingface wav2vec2 model.
            
            if item["noise"] is not None:
                padded_noise[i, :length] = item["noise"]
                
            source_rirs.append(item["source_rir"])
            noise_lengths.append(item["noise_length"])
            noise_start_idxs.append(item["noise_start_idx"])
            noise_rirs_list.append(item["noise_rirs"])
            snrs.append(item["snr"])

            ctx_masks[i, :nr_of_tokens] = item["ctx_mask"]
            ctx_tgt_masks[i, :, :nr_of_tokens] = item["ctx_tgt_masks"]
            tgt_masks[i, :, :nr_of_tokens] = item["tgt_mask"]
            teacher_padding_mask[i, :nr_of_tokens] = False

        return {"audio": padded_audio, 
                "padding_mask": padding_mask,
                "teacher_padding_mask": teacher_padding_mask,
                "noise" : padded_noise, 
                "source_rir": torch.stack(source_rirs), 
                "noise_rirs": torch.stack(noise_rirs_list), 
                "noise_length": torch.tensor(noise_lengths), 
                "noise_start_idx": torch.tensor(noise_start_idxs), 
                "snr": torch.tensor(snrs),
                "ctx_mask" : ctx_masks, 
                "tgt_mask" : tgt_masks, 
                "ctx_tgt_mask": ctx_tgt_masks}

    def make_web_dataset(self, path: str):
        bound_collate = partial(
            self.custom_collate_fn, 
            token_func=self.token_func, 
            target_masks_per_ctx=self.hparams.target_masks_per_context,
            downsampling_rate = self.resampling_factor
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
                    buffersize=512,
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
            pin_memory=self.hparams.pin_memory, 
            num_workers=self.hparams.num_workers,
        )
        return loader
        