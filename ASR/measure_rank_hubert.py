"""Effective-rank anchor: HuBERT Base on the same dev-clean utterances.

Gives the paper a scale reference ("comparable to HuBERT's X of 768") so a
reviewer can't ask whether SpeechJEPA's rank is actually high. Uses the same
utterance sampling seed as measure_rank_speechjepa.py so the frame sets match.

    python measure_rank_hubert.py --dev-clean-dir LibriSpeech/dev-clean
"""

from __future__ import annotations

import argparse
import json

import torch
import torchaudio

from effective_rank import FrameCollector, dead_dimension_fraction, effective_rank
from measure_rank_speechjepa import list_dev_clean, load_wave


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-clean-dir", default="LibriSpeech/dev-clean")
    ap.add_argument("--num-utterances", type=int, default=500)
    ap.add_argument("--max-frames", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=12342)
    ap.add_argument("--out", default="effective_rank_hubert.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()

    files = list_dev_clean(args.dev_clean_dir, args.num_utterances, args.seed)
    collector = FrameCollector(args.max_frames, len(files), seed=args.seed)

    for i, path in enumerate(files):
        wave = load_wave(path).to(device).unsqueeze(0)          # (1, T)
        layer_feats, _ = model.extract_features(wave)           # list of (1,T,768)
        for li, h in enumerate(layer_feats, start=1):
            collector.add(f"layer_{li:02d}", h[0])
        collector.add("encoder_final", layer_feats[-1][0])
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)} utterances")

    results = {"model": "HUBERT_BASE (torchaudio)", "metrics": {}}
    for key in sorted(collector.keys()):
        Z = collector.matrix(key)
        results["metrics"][key] = {
            "effective_rank": round(effective_rank(Z), 2),
            "dead_dim_fraction": round(dead_dimension_fraction(Z), 4),
            "n_frames": int(Z.shape[0]),
            "dim": int(Z.shape[1]),
        }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    er = results["metrics"]["encoder_final"]["effective_rank"]
    print(f"\nHuBERT Base final-layer effective rank: {er:.1f} / 768")
    print(f"Per-layer results written to {args.out}")


if __name__ == "__main__":
    main()