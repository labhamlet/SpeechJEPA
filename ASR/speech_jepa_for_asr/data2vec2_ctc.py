"""
CTC fine-tuning of a fairseq data2vec 2.0 *audio* checkpoint (encoder only),
mirroring the structure of SpeechJEPAForCTC (same batch keys, same freeze
schedule, same greedy decode / WER logging, same tri-stage LR schedule).

Checkpoints (fairseq, examples/data2vec README):
  Base  (pretrained, Librispeech 960h): https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_libri.pt
  Large (pretrained, Libri-light):      https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_vox.pt

Requirements: fairseq installed *from source* (the pip 0.12.2 release predates
data2vec 2.0), with the repo checkout available so that
`<fairseq_root>/examples/data2vec` can be imported as a user_dir:

    git clone https://github.com/facebookresearch/fairseq
    cd fairseq && pip install -e .

Batch interface is identical to the JEPA module:
    batch["audio"]          (B, T_raw)  float waveform @ 16 kHz
    batch["padding_mask"]   (B, T_raw)  True  = REAL audio sample
    batch["attention_mask"] (B, T_tok)  True  = PADDED token (unused here; the
                                        fairseq model derives its own
                                        feature-level padding mask)
    batch["labels"]         (B, L)      token ids, padded with -100
    batch["text"]           list[str]   reference transcripts
"""

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

from .utils import get_tri_stage_schedule


DATA2VEC2_URLS = {
    "base_libri": "https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_libri.pt",
    "large_vox": "https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_vox.pt",
}


def masked_instance_normalize(audio, mask):
    """Per-utterance zero-mean/unit-var normalization over the non-padded part.

    data2vec 2.0 audio models are trained with task.normalize=True, so this is
    required at fine-tuning time as well.
    """
    active_mask = mask.to(audio.dtype)
    sum_audio = (audio * active_mask).sum(dim=-1, keepdim=True)
    active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = sum_audio / active_count

    variance = (((audio - mean) * active_mask) ** 2).sum(dim=-1, keepdim=True) / active_count
    std = torch.sqrt(variance)

    normalized = (audio - mean) / (std + 1e-5)
    return normalized * active_mask


def load_data2vec2(
    checkpoint_path: str,
    fairseq_root: str,
    mask_prob: float = 0.65,
    mask_length: int = 10,
):
    """Load a data2vec 2.0 audio checkpoint and prepare it for CTC fine-tuning.

    The masking hyper-parameters are written into the audio modality config so
    that calling `model.extract_features(..., mask=True)` performs
    SpecAugment-style time masking (masked spans are zeroed by the model, cf.
    `encoder_zero_mask`), exactly like fairseq's own wav2vec/d2v2 fine-tuning.
    Pretraining-specific mask behavior (inverse masking, prob adjustment,
    mask dropout) is disabled.
    """
    from fairseq import checkpoint_utils
    from fairseq import utils as fairseq_utils

    # data2vec2 (Data2VecMultiModel) lives in examples/data2vec, registered via user_dir
    user_dir = str(Path(fairseq_root) / "examples" / "data2vec")
    fairseq_utils.import_user_module(Namespace(user_dir=user_dir))

    # Flat keys are applied recursively, so these reach model.modalities.audio.*
    arg_overrides = {
        "mask_prob": mask_prob,
        "mask_length": mask_length,
        "mask_prob_adjust": 0.0,
        "inverse_mask": False,
        "mask_dropout": 0.0,
        "keep_masked_pct": 0.0,
        "mask_noise_std": 0.0,
        "clone_batch": 1,
        "drop_path": 0.0,
    }

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides=arg_overrides
    )
    model = models[0]

    # Drop EMA teacher, decoders, projection heads
    try:
        model.remove_pretraining_modules(modality="AUDIO")
    except TypeError:
        model.remove_pretraining_modules()

    # Belt-and-braces: make sure the live modality config carries the
    # fine-tuning mask settings even if the override plumbing missed them.
    audio_enc = model.modality_encoders["AUDIO"]
    for k, v in [
        ("mask_prob", mask_prob),
        ("mask_length", mask_length),
        ("mask_prob_adjust", 0.0),
        ("inverse_mask", False),
        ("mask_dropout", 0.0),
        ("keep_masked_pct", 0.0),
        ("mask_noise_std", 0.0),
        ("encoder_zero_mask", True),
    ]:
        try:
            setattr(audio_enc.modality_cfg, k, v)
        except Exception:
            pass

    normalize = getattr(getattr(task, "cfg", task), "normalize", True)
    assert normalize, "expected a data2vec2 audio model trained with normalize=True"

    embed_dim = cfg.model.embed_dim
    return model, embed_dim


class Data2Vec2ForCTC(pl.LightningModule):
    def __init__(
        self,
        pretrained_d2v2,                   # Data2VecMultiModel from load_data2vec2
        encoder_embedding_dim: int,        # 768 (base) / 1024 (large)
        bundle,
        lr: float = 3e-4,
        total_steps: int = 13000,
        freeze_encoder_updates: int = 10000,
        apply_mask: bool = True,           # SpecAugment time masking during training
        dropout: float = 0.0,
        with_decoder: bool = False,        # kept for interface parity; must be False
        use_superb: bool = False,          # not supported for the fairseq model
    ):
        super().__init__()
        assert not with_decoder, "encoder-only ASR: data2vec2 decoder is a pretraining module"
        assert not use_superb, "layer-weighted probing not wired up for the fairseq model"
        self.save_hyperparameters(ignore=["pretrained_d2v2"])

        self.bundle = bundle
        self.labels = self.bundle.get_labels()
        try:
            self.pad_token_id = self.labels.index("-")
        except ValueError:
            print("Warning: '-' not found in labels! Defaulting pad_token_id to 0.")
            self.pad_token_id = 0

        self.model = pretrained_d2v2
        self.apply_mask = apply_mask
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(encoder_embedding_dim, len(self.labels))

        # Separate metrics to prevent training and eval steps from mixing states
        self.train_wer_metric = WordErrorRate()
        self.val_wer_metric = WordErrorRate()
        self.test_wer_metric = WordErrorRate()
        self.val_wer_metric_greedy = WordErrorRate()
        self.test_wer_metric_greedy = WordErrorRate()

        # ------------------------------------------------------------------
        # Freezing. The conv feature extractor (local_encoder) stays frozen
        # for the entire fine-tuning run (feature_grad_mult=0 convention);
        # the rest of the model is frozen for `freeze_encoder_updates` steps.
        # ------------------------------------------------------------------
        self._audio_encoder = self.model.modality_encoders["AUDIO"]
        self._audio_encoder.local_encoder.requires_grad_(False)
        self._audio_encoder.local_encoder.eval()
        for obj in (self._audio_encoder, getattr(self._audio_encoder, "modality_cfg", None)):
            if obj is not None and hasattr(obj, "local_grad_mult"):
                try:
                    obj.local_grad_mult = 0.0
                except Exception:
                    pass

        if self.hparams.freeze_encoder_updates > 0:
            self.model.requires_grad_(False)
            self.model.eval()
            # local encoder stays frozen regardless (already handled above)

        self.decoder_files = download_pretrained_files("librispeech-4-gram")
        self.beam_search_dev = self._setup_torchaudio_decoder(beam_size=50, lm_weight=2.0, word_score=-1.0)
        self.beam_search_test = self._setup_torchaudio_decoder(beam_size=50)

    # ----------------------------------------------------------------------
    def on_train_batch_start(self, batch, batch_idx):
        self._audio_encoder.local_encoder.eval()  # always frozen, always eval
        if self.global_step < self.hparams.freeze_encoder_updates:
            self.model.eval()

        if self.global_step == self.hparams.freeze_encoder_updates:
            self.print(f"Unfreezing encoder at step {self.global_step}!")
            self.model.train()
            self.model.requires_grad_(True)
            # conv feature extractor never unfreezes
            self._audio_encoder.local_encoder.requires_grad_(False)
            self._audio_encoder.local_encoder.eval()

    # ----------------------------------------------------------------------
    def _setup_torchaudio_decoder(self, beam_size, lm_weight=2.0, word_score=-1.0):
        return ctc_decoder(
            lexicon=self.decoder_files.lexicon,
            tokens=[w.lower() for w in self.labels],
            lm=self.decoder_files.lm,
            nbest=1,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            blank_token="-",
            sil_token="|",
        )

    # ----------------------------------------------------------------------
    def forward(self, audio, padding_mask):
        """
        audio:        (B, T_raw)
        padding_mask: (B, T_raw) True = REAL audio (our convention)

        Returns (logits, feat_lengths):
            logits:       (B, T_tok, V)
            feat_lengths: (B,) valid token count per utterance, derived from
                          the model's own downsampled padding mask.
        """
        fs_padding_mask = ~padding_mask.bool()  # fairseq convention: True = PAD
        audio = audio[:, 0, :]
        res = self.model.extract_features(
            source=audio,
            mode="AUDIO",
            padding_mask=fs_padding_mask,
            mask=self.apply_mask and self.training,
            remove_extra_tokens=True,
        )
        x = res["x"]                       # (B, T_tok, C)
        out_pad = res["padding_mask"]      # (B, T_tok) True = PAD, or None

        if out_pad is not None:
            feat_lengths = (~out_pad).sum(-1).to(torch.long)
        else:
            feat_lengths = torch.full(
                (x.size(0),), x.size(1), dtype=torch.long, device=x.device
            )

        logits = self.lm_head(self.dropout(x))
        return logits, feat_lengths

    # ----------------------------------------------------------------------
    def _greedy_decode(self, logits, lengths):
        """Ultra-fast decoding to compute training WER without bogging down the GPU."""
        preds = torch.argmax(logits, dim=-1)  # (B, T)
        predictions = []
        for i in range(preds.size(0)):
            pred = preds[i][: lengths[i]]
            pred = torch.unique_consecutive(pred)
            pred = pred[pred != self.pad_token_id]

            # Replace boundaries with spaces
            raw_text = "".join([self.labels[p] for p in pred]).replace("|", " ")

            # Split and join cleans up any double/triple spaces
            clean_text = " ".join(raw_text.split())

            predictions.append(clean_text)
        return predictions

    # ----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        audio, labels, padding_mask, text_targets = (
            batch["audio"], batch["labels"], batch["padding_mask"], batch["text"]
        )
        logits, input_lengths = self(audio, padding_mask)

        loss = None
        if labels is not None:
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs, flattened_targets, input_lengths, target_lengths,
                    blank=self.pad_token_id, reduction="sum", zero_infinity=True,
                )
                loss = loss / audio.size(0)  # normalize by #sentences ≈ sentence_avg=True

            with torch.no_grad():
                preds = self._greedy_decode(logits.detach().cpu(), input_lengths.cpu())
                wer = self.train_wer_metric(preds, text_targets)

            self.log("train/ctc_loss", loss, prog_bar=True)
            self.log("train/wer_greedy", wer, prog_bar=True, on_step=True)

        return loss

    # ----------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        audio, padding_mask, text_targets = (
            batch["audio"], batch["padding_mask"], batch["text"]
        )

        logits, feat_lengths = self(audio, padding_mask)
        # emissions = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).cpu().contiguous()

        # lengths_int32 = feat_lengths.cpu().to(torch.int32)
        # beam_results = self.beam_search_dev(emissions, lengths_int32)
        # predictions_lm = [" ".join(res[0].words).strip().upper() for res in beam_results]

        lengths_long = feat_lengths.cpu().to(torch.long)
        predictions_greedy = self._greedy_decode(logits.detach().cpu(), lengths_long)

        if batch_idx == 0:
            print(f"\nTargets:             {text_targets[:2]}")
            # print(f"Predictions (LM):    {predictions_lm[:2]}")
            print(f"Predictions (No LM): {predictions_greedy[:2]}")

        # self.val_wer_metric.update(predictions_lm, text_targets)
        self.val_wer_metric_greedy.update(predictions_greedy, text_targets)

    def on_validation_epoch_end(self):
        val_wer_greedy = self.val_wer_metric_greedy.compute()
        self.log("val/wer_greedy", val_wer_greedy, sync_dist=True)
        self.print(f"\n---> [Step {self.global_step}] Validation WER | Greedy: {val_wer_greedy * 100:.2f}% <---")
        self.val_wer_metric_greedy.reset()

    # ----------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        audio, padding_mask, text_targets = (
            batch["audio"], batch["padding_mask"], batch["text"]
        )

        logits, feat_lengths = self(audio, padding_mask)

        lengths_long = feat_lengths.cpu().to(torch.long)
        predictions_greedy = self._greedy_decode(logits.detach().cpu(), lengths_long)

        if batch_idx == 0:
            print(f"\nTargets:             {text_targets[:2]}")
            print(f"Predictions (No LM): {predictions_greedy[:2]}")

        self.test_wer_metric_greedy.update(predictions_greedy, text_targets)

    def on_test_epoch_end(self):
        test_wer_greedy = self.test_wer_metric_greedy.compute()
        self.log("test/wer_greedy", test_wer_greedy, sync_dist=True)
        self.print(f"\n---> [Step {self.global_step}] Test WER | Greedy: {test_wer_greedy * 100:.2f}% <---")
        self.test_wer_metric.reset()
        self.test_wer_metric_greedy.reset()

    # ----------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     betas=(0.9, 0.98),
                                     lr=self.hparams.lr)
        scheduler = get_tri_stage_schedule(optimizer,
                                           final_lr_scale=0.05,
                                           total_steps=self.hparams.total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

