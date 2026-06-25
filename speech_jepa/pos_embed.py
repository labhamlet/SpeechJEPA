import torch.nn as nn
import torch.nn.functional as F 
import torch 
import math 
from typing import Optional

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def __init__(self):
        super().__init__()
        self.act = nn.functional.gelu

    def forward(self, input):
        return self.act(input)

class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 num_conv_pos_embeddings = 128,
                 num_conv_pos_embedding_groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = Wav2Vec2SamePadLayer(num_conv_pos_embeddings)
        self.activation = GELUActivation()

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    


class NormalizedMaskedConvPositionalEmbedding(Wav2Vec2PositionalConvEmbedding):
    def forward(self, hidden_states, mask):
        """
        hidden_states: (B, T, D)
        mask: (B, T) - Boolean mask where True = Valid/Context, False = Masked/Padding
        """

        mask_float = mask.unsqueeze(1).type_as(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2)
        
        hidden_states = hidden_states * mask_float
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)


        k_size = self.conv.kernel_size[0]
        pad = self.conv.padding[0]
        
        weight = torch.ones((1, 1, k_size), device=hidden_states.device, dtype=hidden_states.dtype)
        valid_counts = F.conv1d(mask_float, weight, padding=pad)
        valid_counts = self.padding(valid_counts) # Shape (B, 1, T)
        counts_f32 = valid_counts.float()
        
        scale_factor = torch.where(
            counts_f32 > 0, 
            k_size / (counts_f32 + 1e-8), 
            torch.zeros_like(counts_f32)
        )
        
        hidden_states = hidden_states * scale_factor.to(hidden_states.dtype)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


# https://github.com/facebookresearch/AudioMAE/blob/main/util/pos_embed.py
import numpy as np
import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Create 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        length: sequence length
    
    Returns:
        pos_embed: [length, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    
    pos = np.arange(length, dtype=np.float64)  # (length,)
    out = np.einsum("m,d->md", pos, omega)  # (length, D/2)
    
    emb_sin = np.sin(out)  # (length, D/2)
    emb_cos = np.cos(out)  # (length, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (length, D)
    return emb

    
# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_pos_embed_img2audio(model, checkpoint_model, orig_size, new_size):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size[0], new_size[1]),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_pos_embed_audio(model, checkpoint_model, orig_size, new_size):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            cls_token = pos_embed_checkpoint[:, 0, :].unsqueeze(1)
            pos_tokens = pos_embed_checkpoint[:, 1:, :]  # remove
            pos_tokens = pos_tokens.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            )  # .permute(0, 3, 1, 2)
            # pos_tokens = torch.nn.functional.interpolate(
            #    pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)

            # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens[:, :, : new_size[1], :]  # assume only time diff
            pos_tokens = pos_tokens.flatten(1, 2)
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_patch_embed_audio(
    model,
    checkpoint_model,
    orig_channel,
    new_channel=1,
    kernel_size=(16, 16),
    stride=(16, 16),
    padding=(0, 0),
):
    if orig_channel != new_channel:
        if "patch_embed.proj.weight" in checkpoint_model:
            # aggregate 3 channels in rgb ckpt to 1 channel for audio
            new_proj_weight = torch.nn.Parameter(
                torch.sum(checkpoint_model["patch_embed.proj.weight"], dim=1).unsqueeze(
                    1
                )
            )
            checkpoint_model["patch_embed.proj.weight"] = new_proj_weight
