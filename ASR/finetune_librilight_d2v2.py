import os 
import torchaudio 
import torch 
from speech_jepa_for_asr.jepa_d2v2 import SpeechJEPAForCTC
from speech_jepa_for_asr.bayesian_optimization import optimize_decoding_hyperparameters 

from utils import _get_feat_extract_output_lengths
import pytorch_lightning as pl 
from data_modules_asr.libri_light import LibriLightDataModule
from functools import partial
from pytorch_lightning.callbacks import LearningRateMonitor

import sys 
sys.path.append("/home/gyuksel3/phd/SpeechJEPA")

from wavjepa.jepa_d2v2 import JEPA

from wavjepa.extractors import ConvFeatureExtractor 
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

from pytorch_lightning import seed_everything


manifest_dir = "manifests"

dev_other = os.path.join(manifest_dir, "dev_other.txt")
dev_clean = os.path.join(manifest_dir, "dev_clean.txt")
test_clean = os.path.join(manifest_dir, "test_clean.txt")
test_other = os.path.join(manifest_dir, "test_other.txt")

dev_other_dir = "LibriSpeech/dev-other",
dev_clean_dir = "LibriSpeech/dev-clean",
test_clean_dir = "LibriSpeech/test-clean",
test_other_dir = "LibriSpeech/test-other", 


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
                     train, 
                     train_dir,  
                     use_superb,
                     use_decoder_for_asr):
    

    train = os.path.join(manifest_dir, train)
    audio_token_func = partial(_get_feat_extract_output_lengths, cfg=conv_cfg)

    datamodule = LibriLightDataModule(
        train = train,
        train_dir = train_dir,
        dev_other = dev_other,
        dev_other_dir = dev_other_dir,
        dev_clean = dev_clean,
        dev_clean_dir = dev_clean_dir,
        test_clean = test_clean,
        test_clean_dir = test_clean_dir,
        test_other = test_other, 
        test_other_dir = test_other_dir,
        tokenizer=CharTokenizer(),
        audio_token_func=audio_token_func,
        max_tokens=4_800_000, 
        num_workers=4,
    )

    model = SpeechJEPAForCTC(
        bundle=bundle,
        pretrained_jepa=pretrained_jepa_model,
        audio_token_func=audio_token_func,
        with_decoder=use_decoder_for_asr,
        lr=1e-4, 
        total_steps=13000,
        freeze_encoder_updates=10000,
        use_superb=use_superb
    )

    trainer = pl.Trainer(
        max_steps=13000,
        accelerator="gpu",
        max_epochs=-1,
        precision="32-true",
        val_check_interval=12000,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.dev_other_dataloader())
    
    # best_params = optimize_decoding_hyperparameters(model, 
    #                                                 datamodule.dev_other_dataloader())
    
    # model.beam_search_test = model._setup_torchaudio_decoder(
    #     beam_size=50, 
    #     lm_weight=best_params["alpha"], 
    #     word_score=best_params["beta"]
    # )
    trainer.test(model, dataloaders=[datamodule.dev_clean_dataloader()])

    

if __name__ == "__main__":

    model_path = str(sys.argv[1]) 
    use_decoder_for_asr = str(sys.argv[2]) == "True" 
    use_superb = str(sys.argv[3]) == "True"
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

    model.load_state_dict(new_state_dict, strict=False)

    train_librilight(pretrained_jepa_model=model, 
                    train_dir="librispeech_finetuning/1h", 
                    train= "1h.txt",
                    use_decoder_for_asr=use_decoder_for_asr,
                    use_superb=use_superb,
                    manifest_dir="manifests")
    