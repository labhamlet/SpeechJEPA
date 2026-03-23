import os 
import torchaudio 
import torch 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoFeatureExtractor

from speech_jepa_for_asr.wav2vec2 import HuggingFaceASRForCTC
from data_modules_asr.libri_light_hf import LibriLightDataModule
from pytorch_lightning import seed_everything

import sys 

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = bundle.get_labels()

torch.set_float32_matmul_precision('medium')
seed_everything(42)

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {char: idx for idx, char in enumerate(LABELS)}
        self.id_to_char = {v : k for k,v in self.char_to_id.items()}
        
    def __call__(self, text):
        text = text.upper().replace(' ', '|')
        return [self.char_to_id[c] for c in text if c in self.char_to_id]
    
    def tokens_to_char(self, tokens):
        return[self.id_to_char[t] for t in tokens]
    

def train_librilight(model_name_or_path, 
                     train_data_root, 
                     val_data_root, 
                     test_data_root,
                     manifest_dir):
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
        max_tokens=1_600_000, 
        num_workers=4,
    )

    model = HuggingFaceASRForCTC(
        model_name_or_path=model_name_or_path,
        bundle=bundle,
        lr=0.0001, 
        total_steps=13000,
        freeze_encoder_updates=10000
    )

    # 4. Setup Trainer
    trainer = pl.Trainer(
        max_steps=13000,
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=-1,
        val_check_interval=6000,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    

if __name__ == "__main__":
    # 1. "facebook/wav2vec2-base"
    # 2. "facebook/hubert-base-ls960"
    # 3. "facebook/data2vec-audio-base"
    model_name = str(sys.argv[1])
    
    train_librilight(
        model_name_or_path=model_name, 
        train_data_root="librispeech_finetuning/1h", 
        val_data_root="LibriSpeech/dev-other",
        test_data_root="LibriSpeech/dev-clean",
        manifest_dir="manifests"
    )