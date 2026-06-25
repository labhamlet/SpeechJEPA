from .data_modules.libri import LibriSpeechDataModule 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from SpeechJEPA.wavjepa.jepa_asr import JEPA 
from speech_jepa.extractors import ConvFeatureExtractor 
from speech_jepa.types import TransformerEncoderCFG, TransformerLayerCFG

from speech_jepa_for_asr.jepa import SpeechJEPAForCTC
import torch 

model_path = "/gpfs/work4/0/prjs1338/saved_models_speech_jepa/Data=LibriSpeech/EMA=0.999/EMAEnd=0.99999/EMASteps=100000/MaxBatchSize=12000000/NrGPUs=1/LR=0.0006/LRWarmup=100000/TargetProb=0.2/TargetLen=5/MinContextBlock=5/ContextRatio=0.4/step=250000.ckpt"
conv_cfg = {
    "conv_kernel": [10,3,3,3,3,2,2],
    "conv_stride": [5,2,2,2,2,2,2],
    "convs" : [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
}
weights = torch.load(
    model_path,
    weights_only=False,
    map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

root_dir = "/projects/0/prjs1338/libri"
if __name__ == "__main__":
    
    extractor = ConvFeatureExtractor(
        conv_layers_spec=conv_cfg["convs"],
        in_channels=1,
    )         

    model = JEPA(
            feature_extractor=extractor,
            transformer_encoder_cfg=TransformerEncoderCFG.create(),
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
            transformer_decoder_cfg=TransformerEncoderCFG.create(),
            transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=384),
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
        else:
            new_key = key
        new_state_dict[new_key] = value

    
    model.load_state_dict(new_state_dict, strict=False)    
    asr_model = SpeechJEPAForCTC(pretrained_jepa=model, vocab_size=29)

    datamodule = LibriSpeechDataModule(root_dir=root_dir, batch_size=4)

    accumulation_steps = 27 

    trainer = Trainer(
        max_steps=80000, 
        accumulate_grad_batches=accumulation_steps,
        precision="bf16-mixed",
        gradient_clip_val=1.0, 
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(asr_model, datamodule=datamodule)
    trainer.test(asr_model, datamodule=datamodule)