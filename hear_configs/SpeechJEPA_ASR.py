import torch

from hear_api.runtime import RuntimeSpeechJEPA
import sys 
import wavjepa 
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

sys.modules['sjepa'] = wavjepa

SR = 16000
def load_model(*args, **kwargs):
    weights = None
    if len(args) != 0:
        model_path = args[0]
        weights = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    conv_cfg = {
            "conv_kernel": [10,3,3,3,3,2,2],
            "conv_stride": [5,2,2,2,2,2,2],
            "convs" : [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        }

    transformer_cfg = dict(
            transformer_encoder_cfg=TransformerEncoderCFG.create(),
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
            transformer_decoder_cfg = TransformerEncoderCFG.create(num_layers=12), 
            transformer_decoder_layers_cfg = TransformerLayerCFG.create(d_model = 192, nhead=3),
    )

    model = RuntimeSpeechJEPA(
        in_channels=1,
        weights=weights,
        sr=SR,
        model_size="base",
        conv_cfg = conv_cfg,
        transformer_cfg=transformer_cfg
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
