import torch
import pytest
import sys
sys.path.append("/home/gyuksel2/SpeechJEPA")
from speech_jepa.jepa import JEPA
from speech_jepa.types import TransformerLayerCFG, TransformerEncoderCFG
from speech_jepa.modules import D2vDecoderConfig
from speech_jepa.extractors import ConvFeatureExtractor


def _tiny_model():
    """Small JEPA on CPU/fp32 for exact-ish comparison. No compile, no DDP."""
    torch.manual_seed(0)
    extractor = ConvFeatureExtractor(
        conv_layers_spec=[(64, 10, 5), (64, 3, 2), (64, 3, 2)],
        in_channels=1, depthwise=False, share_weights_over_channels=None,
    )
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(
            d_model=64, nhead=4, activation="gelu",
        ),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        conv_decoder_cfg=D2vDecoderConfig(decoder_dim=64, decoder_layers=2, decoder_groups=4),
        compile_modules=False,
        attn_dropout=0.0, activation_dropout=0.0, hidden_dropout=0.0, layer_drop=0.0,
    )
    return model.eval()   # eval() kills dropout modules regardless of config


def _random_ctx_mask(B, S, ctx_ratio=0.35, seed=1):
    """ctx_mask convention: True = NOT context. Variable count per row."""
    g = torch.Generator().manual_seed(seed)
    is_ctx = torch.rand(B, S, generator=g) < ctx_ratio
    is_ctx[:, 0] = True
    is_ctx[torch.arange(B), (torch.arange(B) % (S - 1)) + 1] = False
    return ~is_ctx


@torch.no_grad()
def test_packed_equals_masked_on_context_positions():
    """The packed encoder must reproduce the full-sequence encoder's outputs
    at every context position. Non-context positions are undefined in the old
    path (they're attended-over garbage), so we only compare context rows."""
    model = _tiny_model()
    B, S, E = 4, 200, model.encoder_embedding_dim

    x = torch.randn(B, S, E)
    ctx_mask = _random_ctx_mask(B, S)          # (B,S) True = NOT context

    old = model.encoder_forward(x, src_key_padding_mask=ctx_mask)   # (B,S,E)
    new = model.encoder_forward_packed(x, ctx_mask)                 # (B,S,E)

    is_ctx = ~ctx_mask                          # (B,S) True = context
    old_ctx = old[is_ctx]                       # (n_ctx_total, E)
    new_ctx = new[is_ctx]

    # fp32, tiny model → tolerances can be tight. Loosen to 1e-3 for bf16.
    assert torch.allclose(old_ctx, new_ctx, atol=1e-4, rtol=1e-4), (
        f"max abs diff = {(old_ctx - new_ctx).abs().max().item():.2e}"
    )


@torch.no_grad()
def test_packed_writes_mask_token_at_noncontext():
    """Non-context positions in the packed output must be exactly the mask token,
    since the decoder relies on that (it re-blends context over mask tokens)."""
    model = _tiny_model()
    B, S, E = 3, 128, model.encoder_embedding_dim
    x = torch.randn(B, S, E)
    ctx_mask = _random_ctx_mask(B, S, seed=7)

    new = model.encoder_forward_packed(x, ctx_mask)
    non_ctx = ctx_mask                          # True = NOT context
    mt = model.mask_token.to(new.dtype).expand(B, S, E)
    assert torch.allclose(new[non_ctx], mt[non_ctx], atol=0.0), \
        "non-context positions are not exactly the mask token"


@torch.no_grad()
def test_pack_context_indices_are_a_permutation():
    """Sanity: scatter is only safe if idx rows contain no duplicates within
    the real (non-pad) region, and pad rows point at valid in-range indices."""
    model = _tiny_model()
    B, S, E = 5, 137, model.encoder_embedding_dim   # non-multiple-of-64 S on purpose
    x = torch.randn(B, S, E)
    ctx_mask = _random_ctx_mask(B, S, S, seed=3)

    _, idx, pad = model.pack_context(x, ctx_mask)
    assert idx.max() < S and idx.min() >= 0, "idx out of range"
    n_ctx = (~ctx_mask).sum(-1)
    for b in range(B):
        real = idx[b, ~pad[b]]                  # real context indices for row b
        assert real.numel() == n_ctx[b].item(), "wrong number of context tokens gathered"
        assert real.unique().numel() == real.numel(), "duplicate context indices"


@torch.no_grad()
@pytest.mark.parametrize("S", [64, 65, 199, 256])   # multiple-of-64 and not
def test_packed_equivalence_various_lengths(S):
    model = _tiny_model()
    B, E = 2, model.encoder_embedding_dim
    x = torch.randn(B, S, E)
    ctx_mask = _random_ctx_mask(B, S, seed=S)
    old = model.encoder_forward(x, src_key_padding_mask=ctx_mask)
    new = model.encoder_forward_packed(x, ctx_mask)
    is_ctx = ~ctx_mask
    assert torch.allclose(old[is_ctx], new[is_ctx], atol=1e-4, rtol=1e-4)