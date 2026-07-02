import sys

import hydra
import pytest
import torch

sys.path.append("/home/gyuksel2/SpeechJEPA")

from data_modules import SSLDataModule
from train import ComponentFactory

SEED = 0
# Short, unequal lengths: keeps the forward passes cheap while still
# exercising padding + bucket logic in custom_collate_fn.
LENGTHS = (32000, 48000, 64000)


def _load_cfg():
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="base",
        )
    return cfg


def _build_model_and_masker(cfg):
    factory = ComponentFactory()
    extractor = factory.create_extractor(cfg)
    model = factory.create_network(cfg, extractor).eval()
    masker = factory.create_masker(cfg)
    return model, masker


def _build_dm(cfg, masker, **overrides):
    kwargs = dict(
        data_dir=cfg.data.data_dir, masker=masker,
        min_sample_len=cfg.data.min_sample_len, max_sample_len=cfg.data.max_sample_len,
        target_batch_size=cfg.data.target_batch_size, max_batch_size=cfg.data.max_batch_size,
        loudness_normalize=cfg.data.loudness_normalize,
        conv_kernel=eval(cfg.extractor.conv_kernel), conv_stride=eval(cfg.extractor.conv_stride),
        target_masks_per_context=cfg.masker.target_masks_per_context,
        bucket_limits=cfg.data.bucket_limits, num_workers=0,
    )
    kwargs.update(overrides)
    return SSLDataModule(**kwargs)


def _run_both_forwards(model, batch):
    audio, ctx, tgt_idx, ctx_tgt, tpad = batch
    with torch.no_grad():
        out_packed = model(audio, ctx, tgt_idx, ctx_tgt, tpad, use_packed=True)
        out_masked = model(audio, ctx, tgt_idx, ctx_tgt, tpad, use_packed=False)
    return out_packed, out_masked


def _assert_equivalence(out_packed, out_masked, tol=1e-4):
    dl = (out_packed["loss"] - out_masked["loss"]).abs().item()
    assert dl < tol, (
        f"loss mismatch: packed={out_packed['loss']:.6f} "
        f"masked={out_masked['loss']:.6f} (delta={dl:.2e})"
    )
    dp = (out_packed["preds"] - out_masked["preds"]).abs().max().item()
    assert dp < tol, f"pred max abs diff = {dp:.2e}"


# --------------------------------------------------------------------------
# Fast path (default): synthetic audio through the real masker + collate.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_batch():
    torch.manual_seed(SEED)
    cfg = _load_cfg()
    model, masker = _build_model_and_masker(cfg)
    dm = _build_dm(cfg, masker)

    # No dm.setup()/dataloader: skips the shuffle(2000) + buffersize=8192
    # pre-fill that dominated runtime. Same _augment_sample, same collate.
    items = [dm._augment_sample({"signal": (torch.randn(L), 16000)}) for L in LENGTHS]
    raw = dm.custom_collate_fn(
        items,
        token_func=dm.token_func,
        target_masks_per_ctx=dm.hparams.target_masks_per_context,
        bucket_limits=dm.bucket_limits,
    )
    batch = model.on_after_batch_transfer(raw, 0)
    # fp32 on CPU: faster than emulated bf16, allows tight tolerances.
    batch = (batch[0].float(),) + tuple(batch[1:])
    return model, batch


@pytest.fixture(scope="module")
def forward_outputs(model_and_batch):
    model, batch = model_and_batch
    return _run_both_forwards(model, batch)


def test_end_to_end_loss_matches(forward_outputs):
    out_packed, out_masked = forward_outputs
    _assert_equivalence(out_packed, out_masked)


def test_no_padded_token_is_context(model_and_batch):
    """Guards the assumption pack_context relies on: padded tokens must be
    marked non-context, else packed/masked diverge on real batches."""
    _, batch = model_and_batch
    _, ctx, _, _, tpad = batch
    overlap = ((~ctx) & tpad).sum().item()
    assert overlap == 0, f"{overlap} padded tokens are marked as context"


def test_pack_context_is_tight(model_and_batch, forward_outputs):
    """On a tiny synthetic batch (S ~ 200 tokens), pad_multiple=64 rounding
    dominates, so a 'saves >40%' economy check is meaningless here (it
    measures rounding granularity, not the masker). Instead assert the
    correctness properties that hold at ANY sequence length:
      1. L is exactly as tight as pad_multiple allows,
      2. no context token is dropped and no non-context token sneaks in.
    The economy check lives in the slow real-shard test below.
    """
    model, batch = model_and_batch
    _, ctx, _, _, _ = batch
    out_packed, _ = forward_outputs
    # Reuse features already computed in the shared forward pass.
    _, idx, pad = model.pack_context(out_packed["local_features"], ctx)

    S = ctx.shape[1]
    L = idx.shape[1]
    n_ctx = (~ctx).sum(-1)                     # context tokens per row
    expected_L = min(S, ((int(n_ctx.max()) + 63) // 64) * 64)
    assert L == expected_L, f"L={L}, expected ceil64(max n_ctx)={expected_L}"

    # Every non-pad packed slot holds a real context token, none are lost.
    assert torch.equal((~pad).sum(-1), n_ctx)
    # The gathered positions for non-pad slots are exactly the context positions.
    for b in range(ctx.shape[0]):
        packed_pos = idx[b][~pad[b]].sort().values
        true_pos = torch.nonzero(~ctx[b]).flatten()
        assert torch.equal(packed_pos, true_pos)


@pytest.fixture(scope="module")
def real_model_and_batch():
    torch.manual_seed(SEED)
    cfg = _load_cfg()
    model, masker = _build_model_and_masker(cfg)
    dm = _build_dm(
        cfg, masker,
        target_batch_size=200_000,   # small batch: one-to-few real clips
        max_batch_size=300_000,
        shuffle_buffer=10,           # vs 2000 default
        bucket_buffersize=32,        # vs 8192 default
    )
    dm.setup("fit")
    raw = next(iter(dm.train_dataloader()))
    batch = model.on_after_batch_transfer(raw, 0)
    batch = (batch[0].float(),) + tuple(batch[1:])
    return model, batch


@pytest.mark.slow
def test_real_shards_end_to_end(real_model_and_batch):
    model, batch = real_model_and_batch
    out_packed, out_masked = _run_both_forwards(model, batch)
    _assert_equivalence(out_packed, out_masked)

    # Padded-token invariant on real data too.
    _, ctx, _, _, tpad = batch
    assert ((~ctx) & tpad).sum().item() == 0

    # Economy check belongs here: real clips are long enough (up to ~780
    # tokens at 250k samples) that the 64-token rounding is negligible and
    # the fraction actually reflects the masker.
    _, idx, _ = model.pack_context(out_packed["local_features"], ctx)
    frac = idx.shape[1] / ctx.shape[1]
    print(f"packed L={idx.shape[1]} vs full S={ctx.shape[1]}  ({frac:.1%})")
    assert frac < 0.6, f"packing kept {frac:.1%} of tokens - barely saving anything"