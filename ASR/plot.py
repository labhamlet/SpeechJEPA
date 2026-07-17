"""Plot effective-rank results from measure_rank_speechjepa.py.

Produces a two-panel, ICASSP-ready figure (PDF + PNG):
  (a) effective rank vs. pre-training updates -- final encoder layer (solid)
      and Eq. (2) teacher targets (dashed) for the default RoPE model and the
      ConvPos variant, with an optional HuBERT Base anchor line;
  (b) per-layer rank profile at the last checkpoint (0 = encoder input).

Usage:
    python plot_rank.py \
        --rope rank_rope_default.json \
        --convpos rank_convpos.json \
        --hubert effective_rank_hubert.json \
        --out figures/effective_rank

Works with single-checkpoint JSONs too (panel (a) shows markers only).
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style: compact, serif, colorblind-safe -- matches typical ICASSP figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "pdf.fonttype": 42,          # embed TrueType -> no Type-3 font complaints
    "ps.fonttype": 42,
})

C_ROPE = "#0173B2"      # blue
C_CONV = "#DE8F05"      # orange
C_HUB = "#555555"       # gray anchor
DIM = 768


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run(path: str) -> list[dict]:
    """Load one model's JSON (list of per-checkpoint dicts), sorted by step."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):        # tolerate a single-entry dict
        data = [data]
    data = [d for d in data if "metrics" in d]
    return sorted(data, key=lambda d: d.get("step") or 0)


def series(run: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for d in run:
        if key in d["metrics"]:
            steps.append(d.get("step") or 0)
            vals.append(d["metrics"][key]["effective_rank"])
    return steps, vals


def layer_profile(entry: dict) -> tuple[list[int], list[float]]:
    """(layer indices, ranks): 0 = embedding, 1..L = transformer layers."""
    mets = entry["metrics"]
    idxs, vals = [], []
    if "embedding" in mets:
        idxs.append(0)
        vals.append(mets["embedding"]["effective_rank"])
    layer_keys = sorted(k for k in mets if re.fullmatch(r"layer_\d+", k))
    for k in layer_keys:
        idxs.append(int(k.split("_")[1]))
        vals.append(mets[k]["effective_rank"])
    return idxs, vals


def hubert_final(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]["encoder_final"]["effective_rank"]


def hubert_layers(path: str) -> tuple[list[int], list[float]] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    mets = data["metrics"]
    layer_keys = sorted(k for k in mets if re.fullmatch(r"layer_\d+", k))
    if not layer_keys:
        return None
    return ([int(k.split("_")[1]) for k in layer_keys],
            [mets[k]["effective_rank"] for k in layer_keys])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", required=True, help="rank_rope_default.json")
    ap.add_argument("--convpos", required=True, help="rank_convpos.json")
    ap.add_argument("--hubert", default="", help="effective_rank_hubert.json (optional)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="effective_rank",
                    help="output path prefix (writes <out>.pdf and <out>.png)")
    ap.add_argument("--width", type=float, default=7.0,
                    help="figure width in inches (7.0 ~ ICASSP \\textwidth; "
                         "use 3.4 with --single-panel for one column)")
    args = ap.parse_args()

    rope = load_run(args.rope)
    conv = load_run(args.convpos)
    tgt_key = f"target_top{args.top_k}"

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(args.width, 2.1),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28},
    )

    # ---- (a) rank vs updates ------------------------------------------------
    for run, color, name in [(rope, C_ROPE, "RoPE (default)"),
                             (conv, C_CONV, "ConvPos")]:
        s, v = series(run, "encoder_final")
        ax_a.plot([x / 1000 for x in s], v, "-o", color=color,
                  label=f"{name}: final layer")
        s, v = series(run, tgt_key)
        ax_a.plot([x / 1000 for x in s], v, "--s", color=color, alpha=0.75,
                  markerfacecolor="white",
                  label=f"{name}: targets (Eq. 2)")

    hub = hubert_final(args.hubert)
    if hub is not None:
        ax_a.axhline(hub, color=C_HUB, lw=0.9, ls=":", zorder=1)
        ax_a.annotate(f"HuBERT Base ({hub:.0f})",
                      xy=(0.02, hub), xycoords=("axes fraction", "data"),
                      va="bottom", fontsize=6.5, color=C_HUB)

    ax_a.axhline(DIM, color="0.85", lw=0.7)
    ax_a.annotate(f"$d={DIM}$", xy=(0.98, DIM), va="top", ha="right",
                  xycoords=("axes fraction", "data"), fontsize=6.5, color="0.5")
    ax_a.set_xlabel("Pre-training updates (k)")
    ax_a.set_ylabel("Effective rank")
    ax_a.set_ylim(0, DIM * 1.06)
    ax_a.set_title("(a) Rank across pre-training", pad=4)
    ax_a.legend(loc="lower right", handlelength=1.8, labelspacing=0.25)
    ax_a.grid(alpha=0.25, lw=0.4)

    # ---- (b) per-layer profile at the last checkpoint ----------------------
    for run, color, name in [(rope, C_ROPE, "RoPE (default)"),
                             (conv, C_CONV, "ConvPos")]:
        idxs, vals = layer_profile(run[-1])
        ax_b.plot(idxs, vals, "-o", color=color, label=name)

    hub_prof = hubert_layers(args.hubert)
    if hub_prof is not None:
        ax_b.plot(hub_prof[0], hub_prof[1], ":d", color=C_HUB,
                  markerfacecolor="white", label="HuBERT Base")

    ax_b.axhline(DIM, color="0.85", lw=0.7)
    ax_b.set_xlabel("Encoder layer (0 = input)")
    ax_b.set_ylabel("Effective rank")
    ax_b.set_ylim(0, DIM * 1.06)
    ax_b.set_xticks(range(0, 13, 2))
    last_step = rope[-1].get("step")
    ax_b.set_title(f"(b) Per-layer rank at {last_step // 1000}k updates"
                   if last_step else "(b) Per-layer rank (final)", pad=4)
    ax_b.legend(loc="lower right", handlelength=1.8, labelspacing=0.25)
    ax_b.grid(alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}.pdf and {args.out}.png")

    # console summary of the numbers cited in the paper text
    for run, name in [(rope, "RoPE"), (conv, "ConvPos")]:
        last = run[-1]
        m = last["metrics"]
        print(f"{name:8s} step={last.get('step')}: "
              f"input={m.get('embedding', {}).get('effective_rank', '-')}, "
              f"final={m['encoder_final']['effective_rank']}, "
              f"targets={m[tgt_key]['effective_rank']}")
        if len(run) > 1:
            _, v = series(run, "encoder_final")
            print(f"{'':8s} final-layer rank range across checkpoints: "
                  f"{min(v):.0f}--{max(v):.0f}")


if __name__ == "__main__":
    main()