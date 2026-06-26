import os
import sys
import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoFeatureExtractor

from speech_jepa_for_asr.wav2vec2 import HuggingFaceASRForCTC
from data_modules_asr.libri_light_hf import LibriLightDataModule

from pytorch_lightning.callbacks import TQDMProgressBar

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = bundle.get_labels()

torch.set_float32_matmul_precision('high')
seed_everything(42)


class OptimizationStepsProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. Update the progress bar's internal step counter (the left side)
        if self.train_progress_bar is not None:
            self.train_progress_bar.n = trainer.global_step
            
            # 2. Extract the latest metrics from the trainer's logged dictionary
            # This captures 'train/ctc_loss', 'lr', etc.
            metrics = trainer.progress_bar_metrics
            
            # 3. Update the description (the right side text) manually
            # This is what TQDM uses to show loss, etc.
            self.train_progress_bar.set_postfix(metrics)
            
            # 4. Refresh only when the step actually increments
            # This prevents the flickering
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


def train_librilight(model_name_or_path, train_data_root, val_data_root,
                     test_data_root, manifest_dir):
    train_manifest = os.path.join(manifest_dir, "1h.txt")
    val_manifest_path = os.path.join(manifest_dir, "dev_other.txt")
    test_manifest_path = os.path.join(manifest_dir, "dev_clean.txt")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

    datamodule = LibriLightDataModule(
        data_root=train_data_root,
        val_data_root=val_data_root,
        test_data_root=test_data_root,
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
        train_manifest_path=train_manifest,
        tokenizer=CharTokenizer(),
        feature_extractor=feature_extractor,
        max_tokens=6_400_000,
        num_workers=6
    )

    model = HuggingFaceASRForCTC(
        model_name_or_path=model_name_or_path,
        bundle=bundle,
        lr=5e-5,
        total_steps=13000,
        freeze_encoder_updates=10000,
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
        max_steps=13000,                     # optimization.max_update: 13000
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=4,           # optimization.update_freq: [4]
        val_check_interval=4000, 
        check_val_every_n_epoch=None,
        
        callbacks=[
            checkpoint_callback,
            progress_bar
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")   # best, not last


if __name__ == "__main__":
    # 1. "facebook/wav2vec2-base"        (group norm  -> no attention_mask)
    # 2. "facebook/hubert-base-ls960"    (group norm  -> no attention_mask)
    # 3. "facebook/data2vec-audio-base"  (check feat_extract_norm)
    model_name = str(sys.argv[1])
    train_librilight(
        model_name_or_path=model_name,
        train_data_root="librispeech_finetuning/1h",
        val_data_root="LibriSpeech/dev-other",
        test_data_root="LibriSpeech/dev-clean",
        manifest_dir="manifests",
    )