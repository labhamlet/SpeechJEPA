import sys

sys.path.append("..")
import torch

from wavjepa.jepa_asr import JEPA as JEPAASR
from SpeechJEPA.wavjepa.jepa import JEPA as JEPA
from wavjepa.extractors import ConvFeatureExtractor
from .feature_helper import FeatureExtractor
from functools import partial 

def instance_normalize(return_audio):
    mean = return_audio.mean(dim=(-2, -1), keepdim=True)
    std = return_audio.std(dim=(-2, -1), keepdim=True)
    normalized = (return_audio - mean) / (std + 1e-5)
    return normalized 

def masked_instance_normalize(audio, mask):
    active_mask = mask.unsqueeze(1) 
    sum_audio = (audio * active_mask).sum(dim=-1, keepdim=True)
    active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = sum_audio / active_count
    
    variance = (((audio - mean) * active_mask) ** 2).sum(dim=-1, keepdim=True) / active_count
    std = torch.sqrt(variance)
    
    normalized = (audio - mean) / (std + 1e-5)
    return normalized * active_mask

def _get_feat_extract_output_lengths(input_lengths, cfg):
    def _conv_out_length(input_length, kernel_size, stride):
        return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

    for kernel_size, stride in zip(cfg["conv_kernel"], cfg["conv_stride"]):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
    return input_lengths

def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    # Ensure it's generated on the correct device
    ts = torch.tensor([step * i for i in range(x_len)], device=x.device).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts


class RuntimeSpeechJEPA(torch.nn.Module):
    BUCKET_BOUNDARIES =[32000, 48000, 64000, 80000, 96000, 112000, 128000, 
                         144000, 160000, 176000, 192000, 208000, 224000, 250000]
    
    def __init__(
        self,
        in_channels,
        weights,
        model_size,
        sr,
        conv_cfg,
        transformer_cfg,
        asr : bool = True,
        **kwargs,
    ) -> None:
        
        super().__init__()
        self.sample_rate = sr
        extractor = ConvFeatureExtractor(
            conv_layers_spec=conv_cfg["convs"],
            in_channels=1,
        )         
        self.token_func = partial(_get_feat_extract_output_lengths, cfg=conv_cfg)
        if asr:
            self.model = JEPAASR(
                feature_extractor=extractor,
                resample_sr=self.sample_rate,
                size=model_size,
                **transformer_cfg,
            )
        else:
            self.model = JEPA(
                feature_extractor=extractor,
                resample_sr=self.sample_rate,
                size=model_size,
                **transformer_cfg,
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

        self.model.load_state_dict(new_state_dict, strict=True)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.unit_frames = 250000
        self.output_steps = self.model.extract_audio.total_patches(self.unit_frames)

        self.model.eval()
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings

    def get_timestamp_embeddings(self, audio):
        B = audio.shape[0]
        audio = self.to_feature(audio)
        input_audio_len = audio.shape[-1]
        
        if audio.ndim != 3:
            raise ValueError(
                "audio input tensor must be 3D with shape (n_sounds, n_channels, num_samples)"
            )
                
        x = self.model.get_audio_representation(audio, attention_padding_mask=None)
        ts = get_timestamps(self.sample_rate, B, input_audio_len, x)
        assert ts.shape[-1] == x.shape[1]
        return x, ts
        
    @classmethod
    def get_bucket_length(cls, length: int) -> int:
        for b in cls.BUCKET_BOUNDARIES:
            if length <= b:
                return b
        return cls.BUCKET_BOUNDARIES[-1]