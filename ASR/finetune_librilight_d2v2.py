import os 
import torchaudio 
import torch 
from speech_jepa_for_asr.jepa_d2v2 import SpeechJEPAForCTC
from speech_jepa_for_asr.bayesian_optimization import optimize_decoding_hyperparameters 

from utils import _get_feat_extract_output_lengths
import pytorch_lightning as pl 
from data_modules_asr.libri_light import LibriLightDataModule
from functools import partial
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import sys 
sys.path.append("/home/gyuksel3/phd/SpeechJEPA")

from wavjepa.jepa_d2v2 import JEPA

from wavjepa.extractors import ConvFeatureExtractor 
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

from pytorch_lightning import seed_everything

import hydra

from omegaconf import DictConfig


manifest_dir = "manifests"

dev_other = os.path.join(manifest_dir, "dev_other.txt")
dev_clean = os.path.join(manifest_dir, "dev_clean.txt")
test_clean = os.path.join(manifest_dir, "test_clean.txt")
test_other = os.path.join(manifest_dir, "test_other.txt")

dev_other_dir = "LibriSpeech/dev-other"
dev_clean_dir = "LibriSpeech/dev-clean"
test_clean_dir = "LibriSpeech/test-clean"
test_other_dir = "LibriSpeech/test-other" 


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
    

def train_librilight(
    pretrained_jepa_model,
    cfg: DictConfig,
    train: str,
    train_dir: str,
    use_superb: bool,
    use_decoder_for_asr: bool,
) -> float:
    """
    Trains the model and returns dev-other WER so Ax can optimise it.
    # #Search params in dev-other
    # best_params = optimize_decoding_hyperparameters(model, 
    #                                                 datamodule.dev_other_dataloader())
    
    # model.beam_search_test = model._setup_torchaudio_decoder(
    #     beam_size=1500, 
    #     lm_weight=best_params["alpha"], 
    #     word_score=best_params["beta"]
    # )
    #Test dev-clean, test-other, dev-other and test-clean with the dev-other params

    """
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
        num_workers=4,
    )
 
    model = SpeechJEPAForCTC(
        bundle=bundle,
        pretrained_jepa=pretrained_jepa_model,
        audio_token_func=audio_token_func,
        with_decoder=use_decoder_for_asr,
        lr=cfg.lr,
        total_steps=cfg.steps,
        freeze_encoder_updates=cfg.freeze_encoder_updates,
        use_superb=use_superb,
        mask_time_prob=cfg.mask_time_prob,
        mask_feature_prob=cfg.mask_feature_prob,
        dropout=cfg.dropout
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
 
    trainer = pl.Trainer(
        max_steps=cfg.steps,
        accelerator="gpu",
        max_epochs=-1,
        precision="bf16-mixed",
        val_check_interval=1000,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ],
        devices=cfg.num_gpus,
        strategy='ddp_find_unused_parameters_true'
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
 
        ckpt_model = SpeechJEPAForCTC.load_from_checkpoint(
            ckpt_path,
            bundle=bundle,
            pretrained_jepa=pretrained_jepa_model,
            audio_token_func=audio_token_func,
            with_decoder=use_decoder_for_asr,
            lr=cfg.lr,
            total_steps=cfg.steps,
            freeze_encoder_updates=cfg.freeze_encoder_updates,
            use_superb=use_superb,
            mask_time_prob=cfg.mask_time_prob,
            mask_feature_prob=cfg.mask_feature_prob,
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
    weights = torch.load(
        cfg.model_path,
        weights_only=False,
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    extractor = ConvFeatureExtractor(
        conv_layers_spec=conv_cfg["convs"],
        in_channels=1,
    )         

    #Drops the layer of a transformer (skip the layer with probability p)
    model = JEPA(
                feature_extractor=extractor,
                transformer_encoder_cfg=TransformerEncoderCFG.create(),
                transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
                resample_sr=16000,
                size="base",
                layer_drop = cfg.layer_drop,
                attn_dropout=cfg.attn_dropout, 
                activation_dropout=cfg.activation_dropout,
                hidden_dropout=cfg.hidden_dropout
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
    return model 

@hydra.main(version_base=None, config_path="./configs", config_name="libri_10h.yaml")
def main(cfg: DictConfig) -> float:
    """
    Returns dev-other WER.  When running with the Ax sweeper this value is
    used as the optimisation objective (minimise).
    """
    model = load_model(cfg)
    dev_other_wer = train_librilight(
        pretrained_jepa_model=model,
        cfg=cfg,
        train_dir=cfg.root_dir,
        train=cfg.manifest,
        use_decoder_for_asr=cfg.use_decoder_for_asr,
        use_superb=cfg.use_superb,
    )
    return dev_other_wer
 
 
if __name__ == "__main__":
    main()


#10min 
#Mask prob channel = 0.008
#Mask prob time = 0.075
#Steps=12000
#Freeze=10000
#LR=4e-4
#Batch_size=4_8M
#32.5	37.4	33.2	37.2

#1H
#Mask prob channel = 0.004
#Mask prob time = 0.075
#Steps=13000
#Freeze=10000
#LR=4e-4
#Batch_size=4_8M
#16	22.3	16.4	22.5

#10H?
#Mask prob channel = 0.004
#Mask prob time = 0.065
#Steps=30000
#Freeze=10000
#LR=4e-4
#Batch_size=6_4M
#8.5  16.3  8.6  16.5