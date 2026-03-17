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


    use_encoder_rope = kwargs.get("use_encoder_rope", "False") == "True"
    use_decoder_rope = str(kwargs.get("use_decoder_rope", "False")) == "True"
    use_kernel_dropout_encoder = kwargs.get("use_kernel_dropout_encoder", "False") == "True"
    use_kernel_dropout_decoder = kwargs.get("use_kernel_dropout_decoder", "False") == "True"

    print(f"Encoder RoPE: {use_encoder_rope}")
    print(f"Encoder RoPE: {use_decoder_rope}")
    print(f"Kernel Dropout Encoder: {use_kernel_dropout_encoder}")
    print(f"Kernel Dropout Decoder: {use_kernel_dropout_decoder}")

    conv_cfg = {
            "conv_kernel": [10,3,3,3,3,2,2],
            "conv_stride": [5,2,2,2,2,2,2],
            "convs" : [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        }

    transformer_cfg = dict(
            transformer_encoder_cfg=TransformerEncoderCFG.create(),
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
    )

    model = RuntimeSpeechJEPA(
        in_channels=1,
        weights=weights,
        sr=SR,
        model_size="base",
        conv_cfg = conv_cfg,
        transformer_cfg=transformer_cfg,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
