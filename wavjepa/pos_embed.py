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
    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Adds absolute positional encoding to the input tensor. """
        # x shape: (B, T, D)
        return self.pe[:, :x.size(1), :].to(x.device)