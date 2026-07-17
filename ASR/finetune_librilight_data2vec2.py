import os
import torchaudio
import torch

from speech_jepa_for_asr.data2vec2_ctc import Data2Vec2ForCTC, load_data2vec2

from utils import _get_feat_extract_output_lengths
import pytorch_lightning as pl
from data_modules_asr.libri_light import LibriLightDataModule
from functools import partial
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from pytorch_lightning import seed_everything

import hydra

from omegaconf import DictConfig


FAIRSEQ_ROOT = "/home/gyuksel2/SpeechJEPA/ASR/fairseq"

manifest_dir = "/home/gyuksel2/SpeechJEPA/ASR/manifests"

dev_other = os.path.join(manifest_dir, "dev_other.txt")
dev_clean = os.path.join(manifest_dir, "dev_clean.txt")
test_other = os.path.join(manifest_dir, "test_other.txt")
test_clean = os.path.join(manifest_dir, "test_clean.txt")

dev_other_dir = "LibriSpeech/dev-other"
dev_clean_dir = "LibriSpeech/dev-clean"
test_other_dir = "LibriSpeech/test-other"
test_clean_dir = "LibriSpeech/test-clean"

torch.set_float32_matmul_precision('high')
seed_everything(1234)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = bundle.get_labels()

# data2vec 2.0 audio uses the identical wav2vec2-style conv front end,
# so the token/length function for the datamodule is unchanged.
conv_cfg = {
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "convs": [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]
}


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {char: idx for idx, char in enumerate(LABELS)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def __call__(self, text):
        # LibriSpeech transcripts use ' ' for spaces, but our label uses '|'
        text = text.upper().replace(' ', '|')
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def tokens_to_char(self, tokens):
        return [self.id_to_char[t] for t in tokens]


class OptimizationStepsProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.train_progress_bar is not None:
            self.train_progress_bar.n = trainer.global_step
            metrics = trainer.progress_bar_metrics
            self.train_progress_bar.set_postfix(metrics)
            if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
                self.train_progress_bar.refresh()


def train_librilight(
    pretrained_d2v2_model,
    encoder_embedding_dim: int,
    cfg: DictConfig,
    train: str,
    train_dir: str,
    use_superb: bool,
    use_decoder_for_asr: bool,
) -> float:
    train_manifest = os.path.join(manifest_dir, train)
    audio_token_func = partial(_get_feat_extract_output_lengths, cfg=conv_cfg)

    datamodule = LibriLightDataModule(
        train=train_manifest,
        train_dir=train_dir,
        dev_other=dev_other,
        dev_other_dir=dev_other_dir,
        dev_clean=dev_clean,
        dev_clean_dir=dev_clean_dir,
        test_clean=test_clean,
        test_clean_dir=test_clean_dir,
        test_other=test_other,
        test_other_dir=test_other_dir,
        tokenizer=CharTokenizer(),
        audio_token_func=audio_token_func,
        max_tokens=cfg.max_tokens,
        num_workers=6,
    )

    model = Data2Vec2ForCTC(
        pretrained_d2v2=pretrained_d2v2_model,
        encoder_embedding_dim=encoder_embedding_dim,
        bundle=bundle,
        lr=cfg.lr,
        total_steps=cfg.steps,
        freeze_encoder_updates=cfg.freeze_encoder_updates,
        apply_mask=cfg.mask_time_prob > 0,
        dropout=cfg.dropout,
        with_decoder=use_decoder_for_asr,
        use_superb=use_superb,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer_greedy",
        mode="min",
        save_top_k=2,
        filename="step={step}-wer={val/wer_greedy:.4f}",
        auto_insert_metric_name=False,
        save_last=True,
        verbose=True,
    )

    progress_bar = OptimizationStepsProgressBar()
    trainer = pl.Trainer(
        max_steps=cfg.steps,
        accelerator="gpu",
        max_epochs=-1,
        precision="bf16-mixed",
        val_check_interval=cfg.check_val_every_n_steps * cfg.acc_grad_batches,
        accumulate_grad_batches=cfg.acc_grad_batches,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
            progress_bar
        ],
        devices=cfg.num_gpus,
        strategy='auto'
    )

    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.dev_other_dataloader(),
    )

    best_checkpoints = checkpoint_callback.best_k_models
    ranked = sorted(best_checkpoints.items(), key=lambda x: x[1])

    all_splits = {
        "dev_other":  datamodule.dev_other_dataloader(),
        "dev_clean":  datamodule.dev_clean_dataloader(),
        "test_clean": datamodule.test_clean_dataloader(),
        "test_other": datamodule.test_other_dataloader(),
    }

    results: dict[str, dict[str, float]] = {}

    for rank, (ckpt_path, val_wer) in enumerate(ranked, start=1):
        print(f"\n{'='*60}")
        print(f"  Evaluating top-{rank} checkpoint  (val/wer_greedy={val_wer:.4f})")
        print(f"  {ckpt_path}")
        print(f"{'='*60}")

        ckpt_model = Data2Vec2ForCTC.load_from_checkpoint(
            ckpt_path,
            pretrained_d2v2=pretrained_d2v2_model,
            encoder_embedding_dim=encoder_embedding_dim,
            bundle=bundle,
            lr=cfg.lr,
            total_steps=cfg.steps,
            freeze_encoder_updates=cfg.freeze_encoder_updates,
            apply_mask=cfg.mask_time_prob > 0,
            dropout=cfg.dropout,
            with_decoder=use_decoder_for_asr,
            use_superb=use_superb,
            weights_only=False,
        )

        ckpt_results: dict[str, float] = {"val_wer_greedy": val_wer.item()}
        for split_name, split_loader in all_splits.items():
            split_out = trainer.test(ckpt_model, dataloaders=[split_loader])
            ckpt_results[split_name] = split_out[0]["test/wer_greedy"]

        results[f"top_{rank}"] = {"checkpoint": ckpt_path, **ckpt_results}

    header = (
        f"{'Rank':<6} {'val_wer':>8} {'dev_other':>10} "
        f"{'dev_clean':>10} {'test_other':>11} {'test_clean':>11}"
    )
    print(header)
    print("-" * len(header))
    for rank, (tag, r) in enumerate(results.items(), start=1):
        print(
            f"{rank:<6} "
            f"{r['val_wer_greedy'] * 100:>7.2f}% "
            f"{r['dev_other'] * 100:>9.2f}% "
            f"{r['dev_clean'] * 100:>9.2f}% "
            f"{r['test_other'] * 100:>10.2f}% "
            f"{r['test_clean'] * 100:>10.2f}%"
        )

    # Best checkpoint WER returned to Ax as the optimisation objective
    best_ckpt_path, best_val_wer = ranked[0]
    print(f"\nBest checkpoint : {best_ckpt_path}  (val/wer_greedy={best_val_wer:.4f})")

    return best_val_wer.item()


def load_model(cfg):
    """
    Replaces the JEPA reconstruction + state-dict surgery. The fairseq .pt
    already contains config + weights, so point `model_path` in the yaml at the
    downloaded checkpoint (e.g. base_libri.pt) - no other config changes needed.
    The architecture blocks in the yaml (transformer_*_cfg, pos_embedding,
    decoder, decoder_groups, layer_drop, attn/activation/hidden dropout) are
    simply unused here: the checkpoint defines its own architecture and
    regularization.

    SpecAugment: `mask_time_prob` is wired into the model's own time masking.
    `mask_feature_prob` is unused (d2v2 has no channel-masking path here; at
    0.004 its effect is negligible anyway).
    """
    print(cfg.model_path)
    model, embed_dim = load_data2vec2(
        checkpoint_path=cfg.model_path,
        fairseq_root=FAIRSEQ_ROOT,
        mask_prob=cfg.mask_time_prob,
        mask_length=10,
    )
    return model, embed_dim


@hydra.main(config_path="./configs", config_name="libri_1h.yaml")
def main(cfg: DictConfig) -> float:
    """
    Returns dev-other WER.  When running with the Ax sweeper this value is
    used as the optimisation objective (minimise).
    """
    model, embed_dim = load_model(cfg)
    dev_other_wer = train_librilight(
        pretrained_d2v2_model=model,
        encoder_embedding_dim=embed_dim,
        cfg=cfg,
        train_dir=cfg.root_dir,
        train=cfg.manifest,
        use_decoder_for_asr=cfg.use_decoder_for_asr,
        use_superb=cfg.use_superb,
    )
    return dev_other_wer


if __name__ == "__main__":
    main()