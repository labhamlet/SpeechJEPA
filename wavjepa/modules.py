import torch
import torch.nn as nn
from torchtune.modules import (
    MultiHeadAttention,
    RotaryPositionalEmbeddings
)
from dataclasses import dataclass
import torch.nn.functional as F

class TorchtuneEncoder(nn.Module):
    def __init__(self, d_model: int,
                 dim_feedforward: int,
                 norm_first: bool, 
                 nhead: int,
                 num_layers: int, 
                 use_rope : bool,
                 max_seq_len=8192):
        
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm_first = norm_first
        
        head_dim = self.d_model // nhead
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiHeadAttention(
                embed_dim=self.d_model,
                num_heads=nhead,
                num_kv_heads=nhead,
                head_dim=head_dim,
                q_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                k_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                v_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                output_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                pos_embeddings=rope if use_rope else None,
                attn_dropout=0.0
            )
            
            mlp = nn.Sequential(
                nn.Linear(self.d_model, dim_feedforward, bias=True),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(dim_feedforward, self.d_model, bias=True),
                nn.Dropout(0.0)
            )
            
            norm_sa = nn.LayerNorm(self.d_model, eps=1e-6)
            norm_mlp = nn.LayerNorm(self.d_model, eps=1e-6)
            
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'mlp': mlp,
                'norm_sa': norm_sa,
                'norm_mlp': norm_mlp
            }))

        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-6) if self.norm_first else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        B, S, E = x.shape
        
        # SDPA Mask: torchtune/SDPA wants True = Keep, False = Mask.
        # JEPA/nn.Transformer wants True = Mask, False = Keep.
        mask = None
        if src_key_padding_mask is not None:
            valid_tokens = ~src_key_padding_mask 
            mask = valid_tokens.view(B, 1, S).expand(B, S, S)

        for layer in self.layers:
            if self.norm_first: # Pre-Norm (Modern/LLM style)
                normed_x = layer['norm_sa'](x)
                x = x + layer['attn'](normed_x, normed_x, mask=mask)
                x = x + layer['mlp'](layer['norm_mlp'](x))
            else: # Post-Norm
                # Pass 'x' twice: once for query, once for key/value
                x = layer['norm_sa'](x + layer['attn'](x, x, mask=mask))
                x = layer['norm_mlp'](x + layer['mlp'](x))

        return self.final_norm(x)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 7
    decoder_layers: int = 4
    input_dropout: float = 0.0

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


class DecoderBase(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig):
        super().__init__()
        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual):
        if (
            residual is None
            or not self.decoder_cfg.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual
        return ret


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)



class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups
        )

    def forward(self, x, mask):
        """
        x: (B, C, T)
        mask: (B, T) - Boolean mask where True = Valid, False = Masked
        """
        mask_float = mask.unsqueeze(1).type_as(x)
        
        x = x * mask_float
        x = self.conv(x)

        k_size = self.conv.kernel_size[0]
        pad = self.conv.padding[0]
        
        weight = torch.ones((1, 1, k_size), device=x.device, dtype=x.dtype)
        valid_counts = F.conv1d(mask_float, weight, padding=pad)
        
        scale_factor = torch.where(
            valid_counts > 0, 
            k_size / (valid_counts + 1e-8), 
            torch.zeros_like(valid_counts)
        )
        
        x = x * scale_factor
        return x


class MaskedDecoderBlock(nn.Module):
    def __init__(self, cfg: D2vDecoderConfig, in_dim: int):
        super().__init__()
        self.conv = MaskedConv1d(
            in_channels=in_dim,
            out_channels=cfg.decoder_dim,
            kernel_size=cfg.decoder_kernel,
            padding=cfg.decoder_kernel // 2,
            groups=cfg.decoder_groups,
        )
        self.pad = SamePad(cfg.decoder_kernel)
        self.transpose_1 = TransposeLast()
        self.ln = LayerNorm(cfg.decoder_dim, elementwise_affine=False)
        self.transpose_2 = TransposeLast()
        self.act = nn.GELU()

    def forward(self, x, mask):
        x = self.conv(x, mask)
        x = self.pad(x)
        x = self.transpose_1(x)
        x = self.ln(x)
        x = self.transpose_2(x)
        x = self.act(x)
        return x


class Decoder1d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__(cfg)

        self.blocks = nn.ModuleList([
                MaskedDecoderBlock(cfg, input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        projs =[]
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        residual = x

        for _, layer in enumerate(self.blocks):
            x = layer(x, mask)
            x = self.add_residual(x, residual)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x
