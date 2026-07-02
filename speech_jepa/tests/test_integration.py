import torch
import pytest
from omegaconf import OmegaConf
import hydra
import sys
sys.path.append("/home/gyuksel2/SpeechJEPA")

from data_modules import SSLDataModule
from train import ComponentFactory   # your factory


def _load_cfg():
    # point at your real training config
    with hydra.initialize(version_base=None, 
                          config_path="../../configs"):
        cfg = hydra.compose(config_name="base")
    return cfg


@pytest.fixture(scope="module")
def model_and_batch():
    cfg = _load_cfg()
    factory = ComponentFactory()
    extractor = factory.create_extractor(cfg)
    model = factory.create_network(cfg, extractor).eval()   # eval() = no dropout

    masker = factory.create_masker(cfg)
    dm = SSLDataModule(
        data_dir=cfg.data.data_dir, masker=masker,
        min_sample_len=cfg.data.min_sample_len, max_sample_len=cfg.data.max_sample_len,
        target_batch_size=cfg.data.target_batch_size, max_batch_size=cfg.data.max_batch_size,
        loudness_normalize=cfg.data.loudness_normalize,
        conv_kernel=eval(cfg.extractor.conv_kernel), conv_stride=eval(cfg.extractor.conv_stride),
        target_masks_per_context=cfg.masker.target_masks_per_context,
        bucket_limits=cfg.data.bucket_limits, num_workers=0,   # 0 = debuggable, no worker hang
    )
    dm.setup("fit")
    raw = next(iter(dm.train_dataloader()))
    # mimic on_after_batch_transfer (CPU version, sr==original_sr assumed for the test)
    batch = model.on_after_batch_transfer(raw, 0)
    return model, batch


@torch.no_grad()
def test_end_to_end_loss_matches(model_and_batch):
    model, batch = model_and_batch
    audio, ctx, tgt_idx, ctx_tgt, tpad = batch

    out_packed = model(audio, ctx, tgt_idx, ctx_tgt, tpad, use_packed=True)
    out_masked = model(audio, ctx, tgt_idx, ctx_tgt, tpad, use_packed=False)

    # bf16 batch → loose tol; tighten if you force fp32
    dl = (out_packed["loss"] - out_masked["loss"]).abs().item()
    assert dl < 1e-2, f"loss mismatch: packed={out_packed['loss']:.5f} masked={out_masked['loss']:.5f} (Δ={dl:.2e})"

    # preds should match too (the stronger check)
    dp = (out_packed["preds"] - out_masked["preds"]).abs().max().item()
    assert dp < 1e-2, f"pred max abs diff = {dp:.2e}"


@torch.no_grad()
def test_no_padded_token_is_context(model_and_batch):
    """Guards the assumption pack_context relies on: padded tokens must be
    marked non-context, else packed/masked diverge on real batches."""
    model, batch = model_and_batch
    _, ctx, _, _, tpad = batch
    is_ctx = ~ctx                    # True = context
    is_pad = tpad                    # True = padded
    overlap = (is_ctx & is_pad).sum().item()
    assert overlap == 0, f"{overlap} padded tokens are marked as context"


@torch.no_grad()
def test_packing_actually_saves_tokens(model_and_batch):
    """Confirms the optimization is real on this batch: packed L should be
    well under full S."""
    model, batch = model_and_batch
    _, ctx, _, _, _ = batch
    _, idx, pad = model.pack_context(model._extract_audio(batch[0]), ctx)
    S = ctx.shape[1]
    L = idx.shape[1]
    frac = L / S
    print(f"packed L={L} vs full S={S}  ({frac:.1%} of sequence)")
    assert frac < 0.6, f"packing kept {frac:.1%} of tokens — barely saving anything"