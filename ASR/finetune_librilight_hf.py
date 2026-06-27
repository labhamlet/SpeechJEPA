import os
import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from transformers import AutoFeatureExtractor

import hydra
from omegaconf import DictConfig

from speech_jepa_for_asr.wav2vec2 import HuggingFaceASRForCTC
from data_modules_asr.libri_light_hf import LibriLightDataModule

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = bundle.get_labels()

torch.set_float32_matmul_precision('high')
seed_everything(42)

manifest_dir = "manifests"

dev_other = os.path.join(manifest_dir, "dev_other.txt")
dev_clean = os.path.join(manifest_dir, "dev_clean.txt")
test_other = os.path.join(manifest_dir, "test_other.txt")
test_clean = os.path.join(manifest_dir, "test_clean.txt")

dev_other_dir = "LibriSpeech/dev-other"
dev_clean_dir = "LibriSpeech/dev-clean"
test_other_dir = "LibriSpeech/test-other"
test_clean_dir = "LibriSpeech/test-clean"


class OptimizationStepsProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. Update the progress bar's internal step counter (the left side)
        if self.train_progress_bar is not None:
            self.train_progress_bar.n = trainer.global_step
            metrics = trainer.progress_bar_metrics
            self.train_progress_bar.set_postfix(metrics)
            if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
                self.train_progress_bar.refresh()


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {char: idx for idx, char in enumerate(LABELS)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def __call__(self, text):
        text = text.upper().replace(' ', '|')
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def tokens_to_char(self, tokens):
        return [self.id_to_char[t] for t in tokens]


def train_librilight(cfg: DictConfig) -> float:
    model_name_or_path = cfg.model_name_or_path
    train_manifest = os.path.join(manifest_dir, cfg.manifest)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

    datamodule = LibriLightDataModule(
        train=train_manifest,
        train_dir=cfg.root_dir,
        dev_other=dev_other,
        dev_other_dir=dev_other_dir,
        dev_clean=dev_clean,
        dev_clean_dir=dev_clean_dir,
        test_clean=test_clean,
        test_clean_dir=test_clean_dir,
        test_other=test_other,
        test_other_dir=test_other_dir,
        tokenizer=CharTokenizer(),
        feature_extractor=feature_extractor,
        max_tokens=cfg.max_tokens,
        num_workers=cfg.num_workers,
    )

    model = HuggingFaceASRForCTC(
        model_name_or_path=model_name_or_path,
        bundle=bundle,
        lr=cfg.lr,
        total_steps=cfg.steps,
        freeze_encoder_updates=cfg.freeze_encoder_updates,
        mask_time_prob=cfg.mask_time_prob,
        mask_time_length=cfg.mask_time_length,
        mask_time_min_masks=cfg.mask_time_min_masks,
        mask_feature_prob=cfg.mask_feature_prob,
        mask_feature_length=cfg.mask_feature_length,
        mask_feature_min_masks=cfg.mask_feature_min_masks,
        layer_drop=cfg.layer_drop,
        activation_dropout=cfg.activation_dropout,
        hidden_dropout=cfg.hidden_dropout,
        attention_dropout=cfg.attention_dropout,
        lm_dropout=cfg.lm_dropout,
    )

    progress_bar = OptimizationStepsProgressBar()
    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer_greedy",
        mode="min",
        save_top_k=2,
        save_last=True,
        every_n_train_steps=1000,            # checkpoint.save_interval_updates: 1000
        filename="step={step}-wer={val/wer_greedy:.4f}",
        auto_insert_metric_name=False,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_steps=cfg.steps,                 # optimization.max_update: 13000
        accelerator="gpu",
        devices=cfg.num_gpus,
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.acc_grad_batches,
        val_check_interval=1000 * cfg.acc_grad_batches,
        check_val_every_n_epoch=None,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
            progress_bar,
        ],
    )

    # Validate on dev-other during training (matches the original val manifest)
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.dev_other_dataloader(),
    )

    # ------------------------------------------------------------------ #
    #  Top-k checkpoint evaluation across all four splits
    # ------------------------------------------------------------------ #
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

        # hparams were saved via save_hyperparameters(), so only `bundle`
        # (which was ignored on save) needs to be supplied here.
        ckpt_model = HuggingFaceASRForCTC.load_from_checkpoint(
            ckpt_path,
            bundle=bundle,
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

    best_ckpt_path, best_val_wer = ranked[0]
    print(f"\nBest checkpoint : {best_ckpt_path}  (val/wer_greedy={best_val_wer:.4f})")

    return best_val_wer.item()


@hydra.main(version_base=None, config_path="./hf_configs", config_name="libri_1h.yaml")
def main(cfg: DictConfig) -> float:
    """
    Fine-tune a HuggingFace SSL model with CTC and return the best dev-other
    (val/wer_greedy). When running with the Ax sweeper this value is used as
    the optimisation objective (minimise).
    """
    return train_librilight(cfg)


if __name__ == "__main__":
    main()