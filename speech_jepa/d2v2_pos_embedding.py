# Add this class to speech_jepa/modules (next to
# NormalizedMaskedConvPositionalEmbedding). It mirrors the audio
# `relative_positional_encoder` from fairseq's data2vec 2.0 AudioEncoder:
#
#   depth x [ Conv1d(k, groups) -> SamePad -> LayerNorm(non-affine, channels)
#             -> GELU ],  k = max(3, width // depth), no weight norm,
#   optional pre-LN (conv_pos_pre_ln).
#
# Masking follows the d2v2.0 convention: invalid positions are ZERO-FILLED
# before the stack and the conv output is NOT count-renormalized (unlike
# NormalizedMaskedConvPositionalEmbedding). Positional signal near mask
# boundaries is therefore attenuated by the zeros in the window, exactly as
# in d2v2.0.

import torch
import torch.nn as nn
import torch.nn.functional as F


class D2v2ConvPositionalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        width: int = 95,   # d2v2.0 audio default (conv_pos_width)
        depth: int = 5,    # d2v2.0 audio default (conv_pos_depth)
        groups: int = 16,  # d2v2.0 audio default (conv_pos_groups)
        pre_ln: bool = False,  # d2v2.0 conv_pos_pre_ln
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = max(3, width // depth)
        # SamePad: for even kernels, conv with padding=k//2 yields T+1 frames;
        # trim the last one. No-op for odd kernels (default k=19 is odd).
        self.num_pad_remove = 1 if self.kernel_size % 2 == 0 else 0

        self.pre_ln = nn.LayerNorm(embed_dim) if pre_ln else None
        self.convs = nn.ModuleList(
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=groups,
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, D)
        mask: (B, T) boolean, True = valid (context for the student,
              non-padded for the teacher). Invalid positions are zero-filled
              before the conv stack (d2v2.0 convention).
        returns positional embeddings of shape (B, T, D) to be ADDED to x.
        """
        if self.pre_ln is not None:
            x = self.pre_ln(x)

        x = x * mask.unsqueeze(-1).type_as(x)

        h = x.transpose(1, 2)  # (B, D, T)
        for conv in self.convs:
            h = conv(h)
            if self.num_pad_remove > 0:
                h = h[..., : -self.num_pad_remove]
            h = h.transpose(1, 2)                      # (B, T, D)
            h = F.layer_norm(h, (self.embed_dim,))     # non-affine, channel dim
            h = F.gelu(h)
            h = h.transpose(1, 2)                      # (B, D, T)

        return h.transpose(1, 2)