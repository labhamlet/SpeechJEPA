import torch
import torch.nn as nn
from torchtune.modules import (
    MultiHeadAttention,
    RotaryPositionalEmbeddings
)

class TorchtuneEncoder(nn.Module):
    def __init__(self, d_model: int,
                 dim_feedforward: int,
                 norm_first: bool, 
                 nhead: int,
                 num_layers: int, 
                 max_seq_len=8192):
        
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm_first = norm_first
        
        head_dim = self.d_model // nhead
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 1. Multi-Head Attention
            attn = MultiHeadAttention(
                embed_dim=self.d_model,
                num_heads=nhead,
                num_kv_heads=nhead,
                head_dim=head_dim,
                q_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                k_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                v_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                output_proj=nn.Linear(self.d_model, self.d_model, bias=True),
                pos_embeddings=rope,
                attn_dropout=0.0
            )
            
            # 2. Feed Forward (MLP)
            mlp = nn.Sequential(
                nn.Linear(self.d_model, dim_feedforward, bias=True),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(dim_feedforward, self.d_model, bias=True),
                nn.Dropout(0.0)
            )
            
            # 3. Norms
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
            # Flip logic: True (padding) becomes False (ignore)
            valid_tokens = ~src_key_padding_mask 
            # SDPA expects [B, 1, S, S] or [B, S, S]. 
            # Expansion is necessary for the fused kernel to trigger correctly.
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