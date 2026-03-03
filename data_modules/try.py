from torch.utils.data import DataLoader

import torch.nn.functional as F 
from wavjepa.masking import SpeechMasker
from functools import partial     

import torch
import torchaudio
import os 

from scene_module.generate_scenes_batch import generate_scene

from NoisySSLDataModule import NoisyMaxTokensBatchSampler, NoisySelfSupervisedSpeechDataset, JEPANaturalisticScene

from try_utils import visualize_masks

def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor | int, cfg):
    """
    Computes the output length of the convolutional layers
    """
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

    for kernel_size, stride in zip(cfg["conv_kernel"], cfg["conv_stride"]):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
    return input_lengths

def fairseq_collate_fn(batch: list[JEPANaturalisticScene], token_func):
    """
    This function now handles padding the audio, the noise, and grouping the metadata.
    """
    batch_size = len(batch)
    lengths = [item.audio.shape[-1] for item in batch]
    max_len = max(lengths)

    padded_audio = torch.zeros(batch_size, max_len)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_noise = torch.zeros(batch_size, max_len)

    #We know that resampling factor is 2, we should use the dataloader class here somehow.
    nr_of_tokens_per_padded_audio = token_func(max_len // 2).item()
    #We should use the cfg here somehow.
    target_masks_per_ctx = 4

    #Initialzie everything as being masked.
    ctx_masks = torch.ones(batch_size, nr_of_tokens_per_padded_audio, dtype=torch.bool)
    ctx_tgt_masks = torch.ones(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype=torch.bool)
    #False means that we do not propagate loss through these target indices.
    tgt_masks = torch.zeros(batch_size, target_masks_per_ctx, nr_of_tokens_per_padded_audio, dtype= torch.bool)
    source_rirs, noise_lengths, noise_start_idxs, noise_rirs_list, snrs = [], [], [], [],[]


    for i, item in enumerate(batch):        
        nr_of_tokens = item.ctx_mask.shape[-1]
        assert nr_of_tokens_per_padded_audio >= nr_of_tokens, "Something went wrong with the padding"

        length = lengths[i]

        padded_audio[i, :length] = item.audio
        padding_mask[i, :length] = True # Mark actual audio frames as True, this is from huggingface wav2vec2 model.
        
        if item.noise is not None:
            padded_noise[i, :length] = item.noise
            
        source_rirs.append(item.source_rir)
        noise_lengths.append(item.noise_length)
        noise_start_idxs.append(item.noise_start_idx)
        noise_rirs_list.append(item.noise_rirs)
        snrs.append(item.snr)

        ctx_masks[i, :nr_of_tokens] = item.ctx_mask
        ctx_tgt_masks[i, :, :nr_of_tokens] = item.ctx_tgt_masks
        tgt_masks[i, :, :nr_of_tokens] = item.tgt_mask

    return (padded_audio, padding_mask, padded_noise, 
            torch.stack(source_rirs), torch.stack(noise_rirs_list), 
            torch.tensor(noise_lengths), torch.tensor(noise_start_idxs), torch.tensor(snrs),
            ctx_masks, tgt_masks, ctx_tgt_masks)



def save_audio(audios, path):
    # Loop through the batch dimension (B)
    os.makedirs(path, exist_ok=True)
    for i in range(audios.shape[0]):
        path_ = os.path.join(path, f"clean_batch_sample{i}.wav")
        audio_to_save = audios[i].detach().cpu()
        max_val = torch.abs(audio_to_save).max()
        if max_val > 0:
            audio_to_save = audio_to_save / max_val * 0.99
        torchaudio.save(
            path_, 
            audio_to_save,
            16000,
        )

def resample(audio: torch.Tensor, resample_sr: int, original_sr=32000) -> torch.Tensor:
    """
    Resample the audio using kaiser best resampling
    """
    return torchaudio.functional.resample(
        audio,
        original_sr,
        resample_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


if __name__ == "__main__":
    cfg = {
        "manifest_path": "/projects/0/prjs1338/libri/maifest.tsv",
        "noise_path": "/projects/0/prjs1338/noise/wham_noise/tr/webdataset_output_32000/shard-{000000..000019}.tar",
        "rir_path": "/gpfs/work5/0/prjs1261/SequentialAmbisonicRIRs32000/train/shard-{000000..000069}.tar",
        "min_sample_size": 32000,
        "max_sample_size": 250000,
        "sr": 16000, 
        "resample_sr": 32000,
        "max_tokens": 1_400_000,
        "loudness_normalize": True,
        "with_rir": True,
        "with_noise": True,
        "conv_kernel": [10,3,3,3,3,2,2],
        "conv_stride": [5,2,2,2,2,2,2],
        "conv_dim" : [512,512,512,512,512,512,512]
    }
    
    masker = SpeechMasker(
                target_masks_per_context=4,
                target_prob=0.2,
                target_length=10,
                ratio_cutoff=0.5,
                channel_based_masking=False,
                min_context_len = 5
                )
    
    token_func = partial(_get_feat_extract_output_lengths, cfg=cfg)
    dataset = NoisySelfSupervisedSpeechDataset(
        manifest_path=cfg["manifest_path"], 
        masker = masker, 
        token_func=token_func,
        rir_path=cfg["rir_path"],
        noise_path=cfg["noise_path"],
        min_sample_size=cfg["min_sample_size"],   
        max_sample_size=cfg["max_sample_size"],  
        sr=cfg["sr"],
        resample_sr=cfg["resample_sr"], 
        loudness_normalize=cfg["loudness_normalize"],
        with_rir=cfg["with_rir"],
        with_noise=cfg["with_noise"]
    )

    batch_sampler = NoisyMaxTokensBatchSampler(
        dataset=dataset, 
        max_tokens=cfg["max_tokens"], 
        shuffle=True
    )


    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(fairseq_collate_fn, token_func=token_func),
        num_workers=16,
        pin_memory=True
    )


    extractor = ConvFeatureExtractor(
            conv_layers_spec=eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"),
            in_channels=1,
    )    

    for batch in dataloader:
        _, _, ctx_masks, tgt_masks, _ = simulate_jepa_training(batch)
        os.makedirs("masks", exist_ok=True)
        i = 0
        for ctx_mask, tgt_mask in zip(ctx_masks, tgt_masks):
            visualize_masks(
                n_times=ctx_mask.shape[-1],
                n_targets_per_context=4,
                in_channels=1,
                ctx_mask = ctx_mask.unsqueeze(0), 
                tgt_mask = tgt_mask.unsqueeze(0),
                save_path=f"masks/{i}"
            )
            i += 1 
        print("pass")