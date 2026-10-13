# Speech-JEPA

**A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Speech Representations**

Speech-JEPA learns speech representations by predicting, in latent space, the representations of
masked spans of a waveform from the visible context.

The released checkpoints are listed in [Pretrained models](#pretrained-models). ASR fine-tuning on
the Libri-light limited-resource splits are included.

---


## Installation

Python ≥ 3.10 and a CUDA GPU are assumed (the code is tested with PyTorch 2.7 / CUDA 12.6).

```bash
git clone https://github.com/labhamlet/SpeechJEPA.git
cd SpeechJEPA

# pre-training environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # or: uv sync  (uses pyproject.toml / uv.lock)

# extras for ASR fine-tuning / inference (torchtune, flashlight-text, transformers, huggingface_hub, …)
pip install -r ASR/req.txt

# extras for HEAR evaluation
pip install -r requirements_eval.txt
```

Using the released models only needs `pip install transformers torch`.

## Pretrained models

All models take 16 kHz mono audio and are published on the Hugging Face Hub in the same layout as the
wav2vec 2.0 checkpoints: they load with `transformers` (`AutoModel` / `AutoModelForCTC`,
`trust_remote_code=True`) and a `Wav2Vec2FeatureExtractor` / `Wav2Vec2Processor`.

| Model | Labelled data | Hub repo | Notes |
|---|---|---|---|
| Speech-JEPA base | none (SSL) | [`labhamlet/speech-jepa-base`](https://huggingface.co/labhamlet/speech-jepa-base) | 370 k updates on LibriSpeech 960 h, 4×H100 |


## Quick start

### Extract representations with the pre-trained model

```python
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("labhamlet/speech-jepa-base")
model = AutoModel.from_pretrained("labhamlet/speech-jepa-base", trust_remote_code=True).eval()

waveform = ...                                              # 1-D float array, 16 kHz
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)

frames = outputs.last_hidden_state      # (1, T // 320, 768): one vector every 20 ms
layers = outputs.hidden_states          # 13 × (1, T // 320, 768): input embedding + every layer
```

Pass a list of waveforms with `padding=True` for batches; the returned `attention_mask` is used to ignore
padded samples. A complete walk-through (batching, pooling, timestamps) is in
[`notebooks/speech_representations.ipynb`](notebooks/speech_representations.ipynb).

The same repository can be loaded back into the training code (decoder and EMA teacher included),
e.g. to fine-tune or continue pre-training:

```python
from wavjepa.hub import load_pretrained

model = load_pretrained("labhamlet/speech-jepa-base")          # wavjepa.jepa_d2v2.JEPA
model = load_pretrained("labhamlet/speech-jepa-base", layer_drop=0.05, attn_dropout=0.1)
```

### Transcribe speech with a fine-tuned model

```python
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("labhamlet/speech-jepa-base-asr-10h")
model = AutoModelForCTC.from_pretrained("labhamlet/speech-jepa-base-asr-10h", trust_remote_code=True).eval()

inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits
print(processor.batch_decode(logits.argmax(-1)))             # ['HELLO WORLD']
```

```bash
python ASR/transcribe.py --model labhamlet/speech-jepa-base-asr-10h a.flac b.wav
```

## Pre-training

### Data

Pre-training reads WebDataset tar shards in which every sample has a `flac` key
(`data_modules/LibriDataModule.py`). Utterances are bucketed by length (2 s – 15.6 s) and packed
into batches of up to `data.max_batch_size` samples (6 M samples ≈ 375 s of audio per GPU).
Point `data.data_dir` at your shards with a brace pattern, e.g.
`"/data/libri/shard-{000000..000112}.tar"`.

### Training

Training is driven by [Hydra](https://hydra.cc) configs in `configs/` and PyTorch Lightning:

```bash
python train_d2v2.py \
    data=librispeech masker=LibriSpeechASR \
    "data.data_dir='/data/libri/shard-{000000..000112}.tar'" \
    trainer.num_gpus=4 save_dir=/path/to/checkpoints
```

The default recipe (`configs/trainer/default_trainer.yaml`, `configs/optimizer/adamW.yaml`,
`configs/masker/LibriSpeechASR.yaml`) is:

| Setting | Value |
|---|---|
| Updates / warm-up | 375 k / 100 k, cosine decay |
| Optimiser | AdamW, lr 4e-4, β = (0.9, 0.98), weight decay 0.01 (not applied to biases / norms) |
| EMA teacher | 0.999 → 0.99999 over 100 k updates |
| Targets | instance-normalised average of the top-8 teacher layers |
| Masking | 4 target blocks per context, target prob 0.15, block length 10 frames, min context block 5, context ratio 0.5 |
| Precision | bf16 mixed, gradient clipping 5.0 |

Checkpoints are written every 5 k updates to
`<save_dir>/saved_models_speech_jepa_d2v2_like/<run identity>/step=<N>.ckpt`.

## ASR fine-tuning

Fine-tuning adds a linear CTC head on top of the encoder, with wav2vec 2.0 style SpecAugment
masking of the encoder input, a tri-stage learning-rate schedule and the encoder frozen for the
first 10 k updates (`ASR/speech_jepa_for_asr/jepa_d2v2.py`).

### Data

Download LibriSpeech (`dev-clean`, `dev-other`, `test-clean`, `test-other`, and `train-clean-100`
for the 100 h setting) and the Libri-light limited-resource supervision set, then create the
tab-separated manifests (`<relative path>\t<num samples>\t<transcript>`):

```bash
cd ASR
# expects LibriSpeech/ and librispeech_finetuning/ (Libri-light 1h/{0..5}, 9h) next to the scripts
python create_libri_manifest.py          # dev/test (+ train-clean-100) manifests
python create_librilight_manifest.py     # 10min / 1h / 10h manifests
```

Ready-made manifests are in `ASR/manifests/`.

### Training

```bash
cd ASR
python finetune_librilight_d2v2.py --config-name libri_10min.yaml   # also libri_1h / libri_10h / libri_100h
```


| Split | Updates | Peak lr | Time / feature mask prob |
|---|---|---|---|---|
| 10 min | 13 k | 1e-4 | 0.075 / 0.008 |
| 1 h | 13 k | 1e-4 | 0.075 / 0.004 |
| 10 h | 20 k | 2e-5  | 0.065 / 0.004 |
| 100 h | 50 k | 2e-5  | 0.05 / 0.008 |

Other entry points: `finetune_librilight_hf.py` (Hugging Face `datasets` loader),
`finetune_librilight_superb.py` (SUPERB-style weighted layer sum),
`speech_jepa_for_asr/bayesian_optimization.py` (Ax search over beam-search LM weights).

## Publishing checkpoints

`hub/export_checkpoint.py` converts a Lightning checkpoint into a Hub repository folder (model code,
weights, processor files and model card) and `hub/publish.py` uploads it:

```bash
hf auth login
python hub/export_checkpoint.py --kind pretrained --include-teacher \
    --ckpt /path/to/step=370000.ckpt --out export/speech-jepa-base --repo-id labhamlet/speech-jepa-base
python hub/publish.py export/speech-jepa-base:labhamlet/speech-jepa-base
```

For a CTC model use `--kind asr --split 10h --results <run dir>/results.json`.


The code is released under the BSD-3-Clause license (see `LICENSE`); the JEPA training loop builds on
[PierreGtch's JEPA implementation](https://github.com/PierreGtch) and the decoder follows
[data2vec 2.0](https://arxiv.org/abs/2212.07525) (fairseq, MIT).
