import os 
import torchaudio 
import torch 
from speech_jepa_for_asr.jepa_superb_like import SpeechJEPAForCTC
from utils import _get_feat_extract_output_lengths
import pytorch_lightning as pl 
from data_modules_asr.libri_light import LibriLightDataModule
from functools import partial
from pytorch_lightning.callbacks import LearningRateMonitor

import sys 
sys.path.append("/home/gyuksel3/phd/SpeechJEPA")

from speech_jepa.jepa import JEPA

from speech_jepa.extractors import ConvFeatureExtractor 
from speech_jepa.types import TransformerEncoderCFG, TransformerLayerCFG

from pytorch_lightning import seed_everything



torch.set_float32_matmul_precision('medium')
seed_everything(42)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
LABELS = bundle.get_labels()
conv_cfg = {
    "conv_kernel":[10,3,3,3,3,2,2],
    "conv_stride": [5,2,2,2,2,2,2],
    "convs" : [(512, 10, 5)] +[(512, 3, 2)] * 4 + [(512,2,2)] +[(512,2,2)]
}

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {char: idx for idx, char in enumerate(LABELS)}
        self.id_to_char = {v : k for k,v in self.char_to_id.items()}
        
    def __call__(self, text):
        # LibriSpeech transcripts use ' ' for spaces, but our label uses '|'
        text = text.upper().replace(' ', '|')
        return [self.char_to_id[c] for c in text if c in self.char_to_id]
    
    def tokens_to_char(self, tokens):
        return[self.id_to_char[t] for t in tokens]
    

def train_librilight(pretrained_jepa_model, 
                     train_data_root, 
                     val_data_root, 
                     test_data_root,
                     manifest_dir,
                     use_decoder_for_asr):
    train_manifest = os.path.join(manifest_dir, "1h.txt")
    val_manifest_path = os.path.join(manifest_dir, "dev_other.txt")
    test_manifest_path = os.path.join(manifest_dir, "test_clean.txt")

    audio_token_func = partial(_get_feat_extract_output_lengths, cfg=conv_cfg)
    datamodule = LibriLightDataModule(
        data_root=train_data_root, 
        val_data_root=val_data_root,
        val_manifest_path=val_manifest_path,
        test_data_root = test_data_root, 
        test_manifest_path = test_manifest_path,
        train_manifest_path=train_manifest,
        tokenizer=CharTokenizer(),
        audio_token_func=audio_token_func,
        max_tokens=1_600_000, 
        num_workers=4,
    )

    model = SpeechJEPAForCTC(
        bundle=bundle,
        pretrained_jepa=pretrained_jepa_model,
        audio_token_func=audio_token_func,
        with_decoder=use_decoder_for_asr,
        lr=0.0001, 
        total_steps=13000,
        freeze_encoder_updates=10000,
    )

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

    model_path = str(sys.argv[1]) 
    use_decoder_for_asr = str(sys.argv[2]) == "True" 

    print(f"Loading Model: {model_path}")
    weights = torch.load(
        model_path,
        weights_only=False,
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    extractor = ConvFeatureExtractor(
        conv_layers_spec=conv_cfg["convs"],
        in_channels=1,
    )         


    model = JEPA(
                feature_extractor=extractor,
                transformer_encoder_cfg=TransformerEncoderCFG.create(),
                transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
                resample_sr=16000,
                size="base",
            )

    new_state_dict = {}
    for key, value in weights["state_dict"].items():
        if key.startswith("extract_audio._orig_mod"):
            new_key = key.replace("extract_audio._orig_mod", "extract_audio")
        elif key.startswith("encoder._orig_mod"):
            new_key = key.replace("encoder._orig_mod", "encoder")
        elif key.startswith("decoder._orig_mod"):
            new_key = key.replace("decoder._orig_mod", "decoder")
        elif key.startswith("teacher_encoder._orig_mod"):
            new_key = key.replace("teacher_encoder._orig_mod", "teacher_encoder")
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)

    train_librilight(pretrained_jepa_model=model, 
                    train_data_root="librispeech_finetuning/1h", 
                    val_data_root="LibriSpeech/dev-other",
                    test_data_root="LibriSpeech/test-clean",
                    use_decoder_for_asr=use_decoder_for_asr,
                    manifest_dir="manifests")
    