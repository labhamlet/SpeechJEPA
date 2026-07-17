"""
CTC fine-tuning of the fairseq DinoSR checkpoint (encoder only), mirroring
the structure of SpeechJEPAForCTC / Data2Vec2ForCTC (same batch keys, same
freeze schedule, same greedy decode / WER logging, same tri-stage LR
schedule).

Checkpoint (official repo, Alexander-H-Liu/dinosr README):
  Base (pretrained, LibriSpeech 960h):
      https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt

Requirements: fairseq installed *from source* (as for data2vec 2.0), plus the
DinoSR repo plugged into fairseq's examples directory so it can be imported
as a user_dir (this is the official instruction: DinoSR registers its model
via `common.user_dir=examples/dinosr`):

    git clone https://github.com/facebookresearch/fairseq
    cd fairseq && pip install -e .
    git clone https://github.com/Alexander-H-Liu/dinosr examples/dinosr
    wget https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt

NOTE (API difference vs. data2vec 2.0): DinoSR's model class follows the
wav2vec2 / data2vec-*audio* template, NOT the multimodal Data2VecMultiModel.
Consequences handled below:
  * no `modality_encoders["AUDIO"]` -- the conv front end is
    `model.feature_extractor`, frozen via `feature_grad_mult = 0`;
  * `extract_features(source, padding_mask, mask=...)` has no
    `mode=`/`remove_extra_tokens=` arguments;
  * masking hyper-parameters are top-level model-config keys
    (`mask_prob`, `mask_length`, `mask_channel_prob`, ...), so the
    fine-tuning overrides are flat wav2vec2-style keys.

Batch interface is identical to the JEPA / d2v2 modules:
    batch["audio"]          (B, T_raw)  float waveform @ 16 kHz
    batch["padding_mask"]   (B, T_raw)  True  = REAL audio sample
    batch["attention_mask"] (B, T_tok)  True  = PADDED token (unused here;
                                        the fairseq model derives its own
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


DINOSR_URLS = {
    "base_libri": "https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt",
}


def load_dinosr(
    checkpoint_path: str,
    fairseq_root: str,
    mask_prob: float = 0.65,
    mask_length: int = 10,
):
    """Load the DinoSR checkpoint and prepare it for CTC fine-tuning.

    Masking overrides are written into the (wav2vec2-style) model config so
    that `model.extract_features(..., mask=True)` performs SpecAugment-style
    time masking during fine-tuning, exactly like fairseq's own wav2vec2 /
    data2vec CTC recipes. Channel masking is disabled (cf. the d2v2 module:
    at mask_feature_prob=0.004 its effect is negligible anyway).
    """
    from fairseq import checkpoint_utils
    from fairseq import utils as fairseq_utils

    # DinoSR registers its model/task when imported as a fairseq user_dir.
    user_dir = Path(fairseq_root) / "examples" / "dinosr"
    if not user_dir.exists():
        raise FileNotFoundError(
            f"{user_dir} not found -- clone https://github.com/Alexander-H-Liu/dinosr "
            f"into <fairseq_root>/examples/dinosr (see module docstring)."
        )
    fairseq_utils.import_user_module(Namespace(user_dir=str(user_dir)))

    # Flat wav2vec2-style keys (DinoSR follows the data2vec-audio template).
    arg_overrides = {
        "mask_prob": mask_prob,
        "mask_length": mask_length,
        "mask_selection": "static",
        "mask_other": 0.0,
        "no_mask_overlap": False,
        "mask_channel_prob": 0.0,
        "mask_channel_length": 10,
        "feature_grad_mult": 0.0,   # conv front end frozen (fairseq convention)
        "dropout_input": 0.0,
        "dropout_features": 0.0,
    }

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides=arg_overrides
    )
    model = models[0]

    # Drop the EMA teacher, online-clustering codebooks, and prediction heads.
    # data2vec-audio-style models expose remove_pretraining_modules(); fall
    # back to manual deletion if DinoSR's version lacks pieces of it.
    try:
        model.remove_pretraining_modules()
    except AttributeError:
        for attr in ("ema", "teacher", "codebooks", "heads", "final_proj"):
            if hasattr(model, attr):
                setattr(model, attr, None)

    # Belt-and-braces: make sure the live config carries the fine-tuning mask
    # settings even if the override plumbing missed them (mirrors the d2v2
    # loader). wav2vec2-style models read most of these from attributes
    # copied off cfg at construction time, so set both.
    for obj in (model, getattr(model, "cfg", None)):
        if obj is None:
            continue
        for k, v in [
            ("mask_prob", mask_prob),
            ("mask_length", mask_length),
            ("mask_channel_prob", 0.0),
            ("feature_grad_mult", 0.0),
        ]:
            try:
                setattr(obj, k, v)
            except Exception:
                pass

    # bf16 fix: wav2vec2-style apply_mask writes the float32 `mask_emb`
    # parameter into the feature tensor via index_put
    # (`x[mask_indices] = self.mask_emb`), which requires exactly matching
    # dtypes -- autocast does not cover index_put, so under bf16-mixed this
    # raises "Index put requires the source and destination dtypes match".
    # Keep mask_emb's dtype in sync with the incoming features. Casting
    # .data in place preserves the registered Parameter (optimizer state is
    # created lazily on the first step, after the first forward, so this is
    # safe), and works for any precision (bf16/fp16/fp32).
    if hasattr(model, "mask_emb") and hasattr(model, "apply_mask"):
        _orig_apply_mask = model.apply_mask  # bound method

        def _apply_mask_dtype_safe(x, *args, **kwargs):
            emb = model.mask_emb
            if emb is not None and emb.dtype != x.dtype:
                emb.data = emb.data.to(x.dtype)
            return _orig_apply_mask(x, *args, **kwargs)

        model.apply_mask = _apply_mask_dtype_safe

    # DinoSR follows the data2vec audio recipe (task.normalize=True), so
    # per-utterance normalization is required at fine-tuning time as well --
    # same assumption as the d2v2 module, enforced the same way.
    normalize = getattr(getattr(task, "cfg", task), "normalize", True)
    assert normalize, "expected a DinoSR model trained with normalize=True"

    embed_dim = cfg.model.encoder_embed_dim  # 768 for Base
    return model, embed_dim


class DinoSRForCTC(pl.LightningModule):
    def __init__(
        self,
        pretrained_dinosr,                 # model from load_dinosr
        encoder_embedding_dim: int,        # 768 (base)
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
        assert not with_decoder, "encoder-only ASR: DinoSR has no decoder"
        assert not use_superb, "layer-weighted probing not wired up for the fairseq model"
        self.save_hyperparameters(ignore=["pretrained_dinosr"])

        self.bundle = bundle
        self.labels = self.bundle.get_labels()
        try:
            self.pad_token_id = self.labels.index("-")
        except ValueError:
            print("Warning: '-' not found in labels! Defaulting pad_token_id to 0.")
            self.pad_token_id = 0

        self.model = pretrained_dinosr
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
        # Freezing. The conv feature extractor stays frozen for the entire
        # fine-tuning run (feature_grad_mult=0 convention); the rest of the
        # model is frozen for `freeze_encoder_updates` steps.
        # ------------------------------------------------------------------
        self._feature_extractor = self.model.feature_extractor
        self._feature_extractor.requires_grad_(False)
        self._feature_extractor.eval()
        if hasattr(self.model, "feature_grad_mult"):
            self.model.feature_grad_mult = 0.0

        if self.hparams.freeze_encoder_updates > 0:
            self.model.requires_grad_(False)
            self.model.eval()
            # feature extractor stays frozen regardless (already handled above)

        self.decoder_files = download_pretrained_files("librispeech-4-gram")
        self.beam_search_dev = self._setup_torchaudio_decoder(beam_size=50, lm_weight=2.0, word_score=-1.0)
        self.beam_search_test = self._setup_torchaudio_decoder(beam_size=50)

    # ----------------------------------------------------------------------
    def on_train_batch_start(self, batch, batch_idx):
        self._feature_extractor.eval()  # always frozen, always eval
        if self.global_step < self.hparams.freeze_encoder_updates:
            self.model.eval()

        if self.global_step == self.hparams.freeze_encoder_updates:
            self.print(f"Unfreezing encoder at step {self.global_step}!")
            self.model.train()
            self.model.requires_grad_(True)
            # conv feature extractor never unfreezes
            self._feature_extractor.requires_grad_(False)
            self._feature_extractor.eval()

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
            padding_mask=fs_padding_mask,
            mask=self.apply_mask and self.training,
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

        lengths_long = feat_lengths.cpu().to(torch.long)
        predictions_greedy = self._greedy_decode(logits.detach().cpu(), lengths_long)

        if batch_idx == 0:
            print(f"\nTargets:             {text_targets[:2]}")
            print(f"Predictions (No LM): {predictions_greedy[:2]}")

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