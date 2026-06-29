import os
import warnings
from functools import partial

import numpy as np
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from .dataset_functions import normalize_audio
from .utils import _get_feat_extract_output_lengths  # noqa: F401

BUCKET_BOUNDARIES = [
    32000, 48000, 64000, 80000, 96000, 112000, 128000,
    144000, 160000, 176000, 192000, 208000, 224000, 250000,
]


def _get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def read_manifest(manifest_path: str, root_dir: str = ""):
    with open(manifest_path) as f:
        lines = f.read().splitlines()

    if lines and "\t" not in lines[0]:
        root_dir = lines[0].strip()
        lines = lines[1:]

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            rel_path, length = parts[0], int(parts[1])
            path = os.path.join(root_dir, rel_path) if root_dir else rel_path
            entries.append((path, length))
        except ValueError:
            continue
    return entries


def batch_by_size(indices, sizes, max_tokens, max_sentences=None):
    batches, cur, cur_max = [], [], 0
    for idx in indices:
        s = sizes[idx]
        new_max = s if s > cur_max else cur_max
        too_big = new_max * (len(cur) + 1) > max_tokens
        too_many = max_sentences is not None and len(cur) >= max_sentences
        if cur and (too_big or too_many):
            batches.append(cur)
            cur, cur_max = [idx], s
        else:
            cur.append(idx)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def get_bucket_length(length: int) -> int:
    for b in BUCKET_BOUNDARIES:
        if length <= b:
            return b
    return BUCKET_BOUNDARIES[-1]


def custom_collate_fn(batch, token_func, target_masks_per_ctx, bucket_limits):
    batch_size = len(batch)
    lengths = [item["signal"].shape[-1] for item in batch]
    max_len = max(lengths)
    if bucket_limits:
        max_len = get_bucket_length(max_len)

    padded_audio = torch.zeros(batch_size, max_len)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    nr_of_tokens_per_padded_audio = token_func(max_len).item()
    teacher_padding_mask = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
    ctx_masks = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
    ctx_tgt_masks = torch.ones(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)
    tgt_masks = torch.zeros(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)

    for i, item in enumerate(batch):
        sig = item["signal"]
        length = lengths[i]
        padded_audio[i, :length] = sig
        padding_mask[i, :length] = True

        nr_of_tokens = item["ctx_mask"].shape[-1]
        ctx_masks[i, :nr_of_tokens] = item["ctx_mask"]
        ctx_tgt_masks[i, :, :nr_of_tokens] = item["ctx_tgt_masks"]
        tgt_masks[i, :, :nr_of_tokens] = item["tgt_mask"]
        teacher_padding_mask[i, :nr_of_tokens] = False

    return {
        "audio": padded_audio,
        "padding_mask": padding_mask,
        "teacher_padding_mask": teacher_padding_mask,
        "ctx_mask": ctx_masks,
        "tgt_mask": tgt_masks,
        "ctx_tgt_mask": ctx_tgt_masks,
    }


class InfiniteBucketedAudioDataset(IterableDataset):
    """Yields fully-collated batches infinitely; the dataset itself controls

    batching, sharding, and reshuffling.
    """

    def __init__(
        self,
        entries,
        masker,
        token_func,
        *,
        min_sample_len: int,
        max_sample_len: int,
        loudness_normalize: bool,
        target_masks_per_context: int,
        target_batch_numel: int,
        max_batch_numel: int,
        bucket_limits: bool,
        seed: int = 0,
    ):
        super().__init__()
        self.entries = entries
        self.masker = masker
        self.token_func = token_func
        self.min_sample_len = min_sample_len
        self.max_sample_len = max_sample_len
        self.loudness_normalize = loudness_normalize
        self.target_masks_per_context = target_masks_per_context
        self.target_batch_numel = target_batch_numel
        self.max_batch_numel = max_batch_numel
        self.bucket_limits = bucket_limits
        self.seed = seed

        self.collate_fn = partial(
            custom_collate_fn,
            token_func=self.token_func,
            target_masks_per_ctx=self.target_masks_per_context,
            bucket_limits=self.bucket_limits,
        )

    def _build_batches(self, cycle: int):
        # The random number generator seed progresses deterministically as we cycle
        rng = np.random.default_rng(self.seed + cycle)

        buckets = {}
        for idx, (_, length) in enumerate(self.entries):
            blen = get_bucket_length(min(length, self.max_sample_len))
            buckets.setdefault(blen, []).append(idx)

        all_batches = []
        for blen in sorted(buckets):
            idxs = rng.permutation(buckets[blen]).tolist()
            if self.bucket_limits:
                sizes = {i: blen for i in idxs}
            else:
                sizes = {i: min(self.entries[i][1], self.max_sample_len) for i in idxs}
            all_batches.extend(batch_by_size(idxs, sizes, self.target_batch_numel))

        order = rng.permutation(len(all_batches))
        return [all_batches[i] for i in order]

    def _crop_audio(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.ndim == 1
        if audio.shape[0] > self.max_sample_len:
            max_start = audio.shape[0] - self.max_sample_len
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            audio = audio[start_idx: start_idx + self.max_sample_len]
        return audio

    def _load_and_augment(self, idx: int):
        path, _ = self.entries[idx]
        audio, _sr = torchaudio.load(path)
        if audio.ndim == 2:
            audio = audio[0]

        if self.loudness_normalize:
            audio = normalize_audio(audio)
        audio = self._crop_audio(audio)

        nr_tokens = self.token_func(audio.shape[-1]).item()
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
            "ctx_tgt_masks": ctx_tgt_masks.squeeze(0),
        }

    def __iter__(self):
        cycle = 0
        rank, world_size = _get_dist_info()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        while True:
            batches = self._build_batches(cycle)

            if world_size > 1:
                per_rank = len(batches) // world_size
                batches = batches[rank::world_size][:per_rank]

            batches = batches[worker_id::num_workers]

            for batch_indices in batches:
                samples = []
                for i in batch_indices:
                    try:
                        samples.append(self._load_and_augment(i))
                    except Exception as e:
                        warnings.warn(f"skipping {self.entries[i][0]}: {e}")
                if not samples:
                    continue
                yield self.collate_fn(samples)

            # Move to the next random cycle seamlessly within this stream
            cycle += 1


class SSLDataModule(pl.LightningDataModule):
    BUCKET_BOUNDARIES = BUCKET_BOUNDARIES

    def __init__(
        self,
        masker,
        manifest_path: str,
        root_dir: str = "",
        min_sample_len: int = 32000,
        max_sample_len: int = 250000,
        target_batch_size: int = 6_000_000,
        max_batch_size: int = 6_000_000,
        loudness_normalize: bool = False,
        conv_kernel: list = [10, 3, 3, 3, 3, 2, 2],
        conv_stride: list = [5, 2, 2, 2, 2, 2, 2],
        target_masks_per_context: int = 4,
        bucket_limits: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["masker"])
        self.masker = masker
        cfg_dict = {"conv_kernel": self.hparams.conv_kernel, "conv_stride": self.hparams.conv_stride}
        self.token_func = partial(_get_feat_extract_output_lengths, cfg=cfg_dict)

    def setup(self, stage: str = None):
        if stage in ("fit", None):
            entries = read_manifest(self.hparams.manifest_path, self.hparams.root_dir)
            before = len(entries)
            entries = [e for e in entries if e[1] >= self.hparams.min_sample_len]
            print(f"[SSLDataModule] {len(entries)}/{before} utterances after min-length filter")

            self.audio_train = InfiniteBucketedAudioDataset(
                entries=entries,
                masker=self.masker,
                token_func=self.token_func,
                min_sample_len=self.hparams.min_sample_len,
                max_sample_len=self.hparams.max_sample_len,
                loudness_normalize=self.hparams.loudness_normalize,
                target_masks_per_context=self.hparams.target_masks_per_context,
                target_batch_numel=self.hparams.target_batch_size,
                max_batch_numel=self.hparams.max_batch_size,
                bucket_limits=self.hparams.bucket_limits,
                seed=self.hparams.seed,
            )

    def train_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=self.hparams.pin_memory,
            num_workers=nw,
            persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
        )