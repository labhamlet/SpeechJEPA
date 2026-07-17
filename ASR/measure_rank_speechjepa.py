"""Measure effective rank (RankMe) of SpeechJEPA representations.

Self-configuring version: instead of guessing the encoder's layer-list
attribute, it probes one forward pass with hooks on EVERY submodule, then
auto-selects the per-layer modules (the repeated module class that emits
d_model-sized outputs). torch.compile is disabled via env vars before torch
is imported, and any already-compiled module is unwrapped to its _orig_mod.

Usage:
    python measure_rank_speechjepa.py \
        --ckpt "/path/to/run_dir/step=*.ckpt" \
        --dev-clean-dir LibriSpeech/dev-clean \
        --step-every 50000 --out rank_rope.json
"""

from __future__ import annotations

# --- must run BEFORE importing torch: makes torch.compile a no-op ----------
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# ---------------------------------------------------------------------------

import argparse
import glob
import itertools
import json
import re
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio

from effective_rank import (
    FrameCollector,
    dead_dimension_fraction,
    effective_rank,
    instance_norm_time,
)

sys.path.append("/home/gyuksel2/SpeechJEPA")

from speech_jepa.jepa import JEPA                        # noqa: E402
from speech_jepa.extractors import ConvFeatureExtractor  # noqa: E402
from speech_jepa.modules import D2vDecoderConfig         # noqa: E402

CONV_CFG = {
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "convs": [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)],
}

# If teacher auto-detection picks the wrong attribute, set it here.
TEACHER_ATTR_OVERRIDE: str | None = None


# ---------------------------------------------------------------------------
# Checkpoint loading (standalone copy of load_model, no hydra)
# ---------------------------------------------------------------------------
def _rebuild_conv_decoder_cfg(raw):
    if raw is None:
        return None
    if isinstance(raw, D2vDecoderConfig):
        raw = vars(raw)
    return D2vDecoderConfig(**dict(raw))


def load_jepa(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    hp = ckpt["hyper_parameters"]

    extractor = ConvFeatureExtractor(
        conv_layers_spec=CONV_CFG["convs"], in_channels=1
    )
    decoder_type = hp.get("decoder_type", "conv")
    conv_decoder_cfg = (
        _rebuild_conv_decoder_cfg(hp.get("conv_decoder_cfg"))
        if decoder_type == "conv" else None
    )

    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=hp["transformer_encoder_layers_cfg"],
        transformer_encoder_cfg=hp["transformer_encoder_cfg"],
        conv_decoder_cfg=conv_decoder_cfg,
        decoder_type=decoder_type,
        transformer_decoder_layers_cfg=hp.get("transformer_decoder_layers_cfg"),
        transformer_decoder_cfg=hp.get("transformer_decoder_cfg"),
        use_conv_pos=hp.get("use_conv_pos", False),
        conv_pos_style=hp.get("conv_pos_style", "d2v2"),
        conv_pos_width=hp.get("conv_pos_width", 95),
        conv_pos_depth=hp.get("conv_pos_depth", 5),
        conv_pos_groups=hp.get("conv_pos_groups", 16),
        conv_pos_pre_ln=hp.get("conv_pos_pre_ln", False),
        use_rope=hp.get("use_rope", True),
        size=hp.get("size", "base"),
        resample_sr=16000,
        layer_drop=0.0,
        attn_dropout=0.0,
        activation_dropout=0.0,
        hidden_dropout=0.0,
    )
    state_dict = {k.replace("._orig_mod", ""): v
                  for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    print(f"Loaded {ckpt_path}\n  rope={hp.get('use_rope', True)} "
          f"conv_pos={hp.get('use_conv_pos', False)} decoder={decoder_type}")
    return model


# ---------------------------------------------------------------------------
# Anti-bypass measures
# ---------------------------------------------------------------------------
def unwrap_compiled(module: nn.Module) -> nn.Module:
    """Peel torch.compile OptimizedModule wrappers so hooks + calls hit the
    eager module."""
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def disable_fused_paths() -> None:
    """Force the slow (Python, hook-visible) Transformer path."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        print("MHA fastpath disabled.")
    except AttributeError:
        print("WARNING: set_fastpath_enabled unavailable in this torch "
              "version; relying on enable_nested_tensor=False.")


def prepare_encoder_for_hooks(encoder: nn.Module) -> None:
    for mod in encoder.modules():
        if isinstance(mod, nn.TransformerEncoder):
            mod.enable_nested_tensor = False
        if hasattr(mod, "use_nested_tensor"):
            mod.use_nested_tensor = False


# ---------------------------------------------------------------------------
# Forward pipeline mirrored from SpeechJEPAForCTC (batch = 1, no masking)
# ---------------------------------------------------------------------------
def masked_instance_normalize(audio, mask):
    active_mask = mask.unsqueeze(1)
    sum_audio = (audio * active_mask).sum(dim=-1, keepdim=True)
    active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = sum_audio / active_count
    variance = (((audio - mean) * active_mask) ** 2).sum(dim=-1, keepdim=True) / active_count
    std = torch.sqrt(variance)
    normalized = (audio - mean) / (std + 1e-5)
    return normalized * active_mask


@torch.no_grad()
def build_encoder_input(model, wave: torch.Tensor, device):
    """wave (T,) -> encoder input x (1, N, D) and attention_mask (1, N)."""
    audio = wave.to(device).view(1, 1, -1)
    padding_mask = torch.ones(1, audio.shape[-1], dtype=torch.bool, device=device)
    audio = masked_instance_normalize(audio, padding_mask)

    x = model.extract_audio(audio)
    x = model.audio_feature_norms(x)
    x = model.post_extraction_mapper(x)
    x = model.local_feature_norms(x)

    attention_mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
    if getattr(model, "use_conv_pos", False):
        x = model.conv_pos_norm(x + model.conv_pos_embedding(x, ~attention_mask))
    return x, attention_mask


@torch.no_grad()
def run_encoder(encoder, x, attention_mask) -> torch.Tensor:
    out = encoder(x, src_key_padding_mask=attention_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def as_frames(t: torch.Tensor) -> torch.Tensor:
    """(1,T,D) or (T,1,D) or (T,D) -> (T,D); layout-agnostic at batch=1."""
    return t.reshape(-1, t.shape[-1])


# ---------------------------------------------------------------------------
# Probe: discover which submodules actually run, pick the layer modules
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_and_select_layers(model, encoder, wave, device, label,
                            num_layers: int = 12) -> list[nn.Module]:
    x, attn = build_encoder_input(model, wave, device)
    d_model = x.shape[-1]

    records: list[tuple[int, str, nn.Module]] = []
    order = itertools.count()
    handles = []

    def make_hook(name, mod):
        def hook(m, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out) and out.ndim == 3 and out.shape[-1] == d_model:
                records.append((next(order), name, m))
        return hook

    for name, mod in encoder.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name, mod)))
    run_encoder(encoder, x, attn)
    for h in handles:
        h.remove()

    if not records:
        raise RuntimeError(
            f"[{label}] probe: NO submodule of {type(encoder).__name__} "
            f"produced a (*, *, {d_model}) output during forward, so the "
            "forward executes outside Python modules entirely. "
            "TORCHDYNAMO_DISABLE=1 and the MHA-fastpath disable are already "
            "active, so this points at a custom fused/functional encoder "
            "forward -- inspect speech_jepa's encoder implementation."
        )

    # group fired modules by (class, depth): the layer stack is >=2 distinct
    # modules of the same class at the same (minimal) depth. Grouping by
    # class alone fails when e.g. a single top-level final LayerNorm shares
    # its class with per-layer norms deeper down.
    groups: dict[tuple[str, int], list[tuple[int, str, nn.Module]]] = defaultdict(list)
    for rec in records:
        key = (type(rec[2]).__name__, rec[1].count("."))
        groups[key].append(rec)

    def distinct(items):
        return len({id(m) for _, _, m in items})

    print(f"  [{label}] probe candidates (class@depth: distinct_modules): "
          + ", ".join(f"{c}@{d}: {distinct(v)}"
                      for (c, d), v in sorted(groups.items())))

    repeated = {k: v for k, v in groups.items() if distinct(v) >= 2}
    pool = repeated if repeated else groups
    if not repeated:
        print(f"  [{label}] WARNING: no repeated (class, depth) group found; "
              "falling back to all candidates -- verify the selection below.")
    best_key = min(pool, key=lambda k: (k[1], -distinct(pool[k])))
    best_cls = f"{best_key[0]} (depth {best_key[1]})"

    seen, ordered = set(), []
    for _, name, mod in sorted(pool[best_key]):        # first-fire order
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        ordered.append((name, mod))

    print(f"  [{label}] selected {len(ordered)} layer modules of class "
          f"{best_cls}: {ordered[0][0]} ... {ordered[-1][0]}")

    # Collapse sub-blocks: many encoders build each Transformer layer from k
    # same-class sub-blocks at the same depth (attn-block + ffn-block, or
    # norm1 + norm2), so the probe sees k*num_layers modules. The LAST
    # sub-block of each layer emits the end-of-layer representation (post-FFN
    # residual in pre-norm designs, norm2 in post-norm designs), so keep
    # every k-th module starting from the k-th.
    if num_layers and len(ordered) != num_layers:
        if len(ordered) % num_layers == 0:
            k = len(ordered) // num_layers
            ordered = ordered[k - 1::k]
            print(f"  [{label}] detected {k} sub-blocks per layer -> keeping "
                  f"the last of each: {len(ordered)} layer-end modules "
                  f"({ordered[0][0]} ... {ordered[-1][0]})")
        else:
            raise RuntimeError(
                f"[{label}] detected {len(ordered)} modules but expected a "
                f"multiple of --num-layers={num_layers}. Inspect the probe "
                "candidate list above and adjust --num-layers or the "
                "selection."
            )
    return [m for _, m in ordered]


class LayerTap:
    """Forward hooks on an explicit list of layer modules."""

    def __init__(self, layer_modules: list[nn.Module]):
        self.outputs: list[torch.Tensor] = []
        self.handles = [m.register_forward_hook(self._hook)
                        for m in layer_modules]
        self.n_layers = len(layer_modules)

    def _hook(self, module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        self.outputs.append(out.detach())

    def reset(self):
        self.outputs = []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def encode_utterance(model, encoder, tap: LayerTap, wave, device):
    x, attn = build_encoder_input(model, wave, device)
    embedding = as_frames(x).clone()
    tap.reset()
    final = as_frames(run_encoder(encoder, x, attn))
    layers = [as_frames(h) for h in tap.outputs]
    if len(layers) != tap.n_layers:
        raise RuntimeError(
            f"expected {tap.n_layers} layer outputs, captured {len(layers)} "
            "-- layer modules fired an unexpected number of times."
        )
    return embedding, layers, final


# ---------------------------------------------------------------------------
# Teacher probing
# ---------------------------------------------------------------------------
def find_teacher_encoder(jepa: nn.Module) -> tuple[nn.Module | None, str]:
    names = ([TEACHER_ATTR_OVERRIDE] if TEACHER_ATTR_OVERRIDE else
             ["teacher", "ema", "ema_model", "target_encoder",
              "teacher_encoder", "ema_encoder", "teacher_model"])
    for name in names:
        obj = getattr(jepa, name, None)
        if obj is None:
            continue
        for sub in ("model", "module", "ema_model", "encoder"):
            inner = getattr(obj, sub, None)
            if isinstance(inner, nn.Module):
                obj = inner
                break
        if isinstance(obj, nn.Module) and any(True for _ in obj.parameters()):
            return obj, name
    return None, ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def list_dev_clean(dev_clean_dir: str, n: int, seed: int) -> list[str]:
    files = sorted(glob.glob(os.path.join(dev_clean_dir, "**", "*.flac"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no .flac under {dev_clean_dir}")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(files), generator=g)[:n]
    return [files[i] for i in idx.tolist()]


def load_wave(path: str) -> torch.Tensor:
    wave, sr = torchaudio.load(path)
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)
    return wave.mean(dim=0)


# ---------------------------------------------------------------------------
# Per-checkpoint measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_checkpoint(ckpt_path, files, device, top_k, max_frames, seed,
                       num_layers=12):
    model = load_jepa(ckpt_path, device)

    student_enc = unwrap_compiled(model.encoder)
    if student_enc is not model.encoder:
        print("  student encoder was torch.compile-wrapped; unwrapped to eager.")
    prepare_encoder_for_hooks(student_enc)

    teacher, teacher_attr = find_teacher_encoder(model)
    if teacher is not None:
        teacher = unwrap_compiled(teacher)
        prepare_encoder_for_hooks(teacher)
        print(f"  EMA teacher found at .{teacher_attr} -- used for Eq.(2) targets")
    else:
        print("  WARNING: no EMA teacher found; Eq.(2) targets from the "
              "STUDENT encoder (set TEACHER_ATTR_OVERRIDE if that's wrong).")

    probe_wave = load_wave(files[0])
    student_layer_mods = probe_and_select_layers(
        model, student_enc, probe_wave, device, "student", num_layers)
    student_tap = LayerTap(student_layer_mods)

    if teacher is not None:
        teacher_layer_mods = probe_and_select_layers(
            model, teacher, probe_wave, device, "teacher", num_layers)
        teacher_tap = LayerTap(teacher_layer_mods)
    else:
        teacher_tap = student_tap

    collector = FrameCollector(max_frames=max_frames, num_utts=len(files),
                               seed=seed)

    for i, path in enumerate(files):
        wave = load_wave(path)
        emb, layers, final = encode_utterance(model, student_enc,
                                              student_tap, wave, device)
        collector.add("embedding", emb)
        for li, h in enumerate(layers, start=1):
            collector.add(f"layer_{li:02d}", h)
        collector.add("encoder_final", final)

        if teacher is not None:
            _, t_layers, _ = encode_utterance(model, teacher,
                                              teacher_tap, wave, device)
        else:
            t_layers = layers
        top = t_layers[-top_k:]
        target = torch.stack([instance_norm_time(h) for h in top]).mean(0)
        collector.add(f"target_top{top_k}", target)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} utterances")

    student_tap.remove()
    if teacher_tap is not student_tap:
        teacher_tap.remove()

    results = {"checkpoint": ckpt_path,
               "teacher_used": teacher is not None,
               "num_utterances": len(files),
               "max_frames": max_frames,
               "top_k": top_k,
               "num_layers_detected": student_tap.n_layers,
               "metrics": {}}
    m = re.search(r"step=(\d+)", ckpt_path)
    results["step"] = int(m.group(1)) if m else None

    for key in sorted(collector.keys()):
        Z = collector.matrix(key)
        results["metrics"][key] = {
            "effective_rank": round(effective_rank(Z), 2),
            "dead_dim_fraction": round(dead_dimension_fraction(Z), 4),
            "n_frames": int(Z.shape[0]),
            "dim": int(Z.shape[1]),
        }
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="checkpoint path(s); globs like 'dir/step=*.ckpt' ok")
    ap.add_argument("--dev-clean-dir", default="LibriSpeech/dev-clean")
    ap.add_argument("--num-utterances", type=int, default=500)
    ap.add_argument("--max-frames", type=int, default=25000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=12,
                    help="expected Transformer layer count; if the probe "
                         "finds k*num_layers same-class modules it keeps "
                         "the last of each group of k. 0 disables the "
                         "check and uses all detected modules")
    ap.add_argument("--seed", type=int, default=12342)
    ap.add_argument("--step-every", type=int, default=0,
                    help="keep only checkpoints whose step is a multiple of "
                         "this (e.g. 50000); 0 = keep all matched checkpoints")
    ap.add_argument("--include-last", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="always keep the highest-step checkpoint even if its "
                         "step is not a multiple of --step-every")
    ap.add_argument("--out", default="effective_rank_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    disable_fused_paths()

    ckpts = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints match {args.ckpt}")

    def step_of(path):
        m = re.search(r"step=(\d+)", path)
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=step_of)
    if args.step_every > 0:
        last = ckpts[-1]
        kept = [p for p in ckpts
                if step_of(p) >= 0 and step_of(p) % args.step_every == 0]
        if args.include_last and last not in kept:
            kept.append(last)
        skipped = len(ckpts) - len(kept)
        ckpts = kept
        print(f"step filter (every {args.step_every}): kept "
              f"{[step_of(p) for p in ckpts]}, skipped {skipped} checkpoint(s)")
        if not ckpts:
            raise FileNotFoundError(
                f"no checkpoints left after --step-every {args.step_every}")

    files = list_dev_clean(args.dev_clean_dir, args.num_utterances, args.seed)
    print(f"{len(ckpts)} checkpoint(s), {len(files)} dev-clean utterances\n")

    all_results = []
    for ckpt in ckpts:
        all_results.append(
            measure_checkpoint(ckpt, files, device, args.top_k,
                               args.max_frames, args.seed,
                               num_layers=args.num_layers)
        )
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'step':>8}  {'enc_final':>10}  {'target_top' + str(args.top_k):>12}"
          f"  {'dead%':>6}  checkpoint")
    for r in all_results:
        mets = r["metrics"]
        print(f"{str(r['step']):>8}  "
              f"{mets['encoder_final']['effective_rank']:>10.1f}  "
              f"{mets[f'target_top{args.top_k}']['effective_rank']:>12.1f}  "
              f"{100 * mets['encoder_final']['dead_dim_fraction']:>5.1f}%  "
              f"{r['checkpoint']}")
    print(f"\nFull per-layer results written to {args.out}")


if __name__ == "__main__":
    main()