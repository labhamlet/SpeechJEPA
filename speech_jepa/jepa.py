import copy
import transformers
import torchaudio

from typing import List, Any, Optional

import torch
import torch.distributed as dist


from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
import pytorch_lightning as pl
from speech_jepa.functions import trunc_normal_
from speech_jepa.extractors.audio_extractor import Extractor
from speech_jepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG

from speech_jepa.modules import TorchtuneEncoder, Decoder1d, D2vDecoderConfig
from speech_jepa.pos_embed import NormalizedMaskedConvPositionalEmbedding
from speech_jepa.d2v2_pos_embedding import D2v2ConvPositionalEncoder

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)

def resample(audio: torch.Tensor, resample_sr: int, original_sr : int) -> torch.Tensor:
    """
    Resample the audio using kaiser best resampling
    """
    return torchaudio.functional.resample(
        audio,
        original_sr,
        resample_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def masked_instance_normalize(audio, mask):
    # Mask: True where audio is real, False where it's padded
    active_mask = mask.unsqueeze(1) # Shape: (B, 1, T)
    # Calculate mean using only active elements
    sum_audio = (audio * active_mask).sum(dim=-1, keepdim=True)
    active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = sum_audio / active_count

    variance = (((audio - mean) * active_mask) ** 2).sum(dim=-1, keepdim=True) / active_count
    std = torch.sqrt(variance)

    normalized = (audio - mean) / (std + 1e-5)
    return normalized * active_mask



class JEPA(pl.LightningModule):
    """
    Joint-Embedding Predictive Architecture (JEPA).

    This implementation is inspired by:
        * I-JEPA http://arxiv.org/abs/2301.08243
        * Data2vec 2.0 http://arxiv.org/abs/2212.07525

    Args:
        feature_encoder:
            Does the local feature encoding. Will be shared between teacher and student.

                * Input: dict with keys: ``**batch``
                * Output: ``local_features`` (batch_size, n_patches, emb_dim)

        mask_maker:
            Computes the training masks as indices.

                * Input: dict with keys:
                    - ``**batch``
                    - ``local_features`` (batch_size, n_patches, emb_dim)

                * output: tuple:
                    - ``idxs_context`` (batch_size, self.n_contexts_per_input, n_context_patches)
                    - ``idxs_target`` (batch_size, self.n_contexts_per_input, self.n_targets_per_context, n_target_patches)

        transformer_kwargs:
            Arguments for :class:`nn.Transformer`. The transformer will have the
            following signature:

                * Input: (batch_size, n_context_patches, emb_dim)
                * Output: (batch_size, n_target_patches, emb_dim)

        loss_fn: nn.Module
            Loss function to use between the ``preds`` (output of the transformer)
            and the ``targets``.
        ema_decay: float
            initial ema decay rate.
        ema_end_decay: float
            final ema decay rate.
        ema_anneal_end_step: int
            when to finish annealing ema decay rate.
        average_top_k_layers: int
            The targets are the average of the outputs of the last k layers of
            the teacher encoder. This parameter specifies the number of layers to
            use for the average.
        decoder_type: str
            Ablation switch for the predictor/decoder. ``"conv"`` uses the
            original :class:`Decoder1d`. ``"transformer"`` uses a shallow
            :class:`TorchtuneEncoder` with RoPE over the (context + mask token)
            sequence.
        transformer_decoder_layers_cfg: Optional[TransformerLayerCFG]
            Layer configuration for the transformer decoder, built with
            :meth:`TransformerLayerCFG.create` exactly like the encoder's
            layer cfg (``d_model``, ``nhead``, ``dim_feedforward`` via
            ``mlp_ratio``, ``norm_first``, ``dropout``, ...). Required when
            ``decoder_type == "transformer"``. If the decoder ``d_model``
            differs from the encoder embedding dim, linear input/output
            projections are added automatically (identity otherwise).
        transformer_decoder_cfg: Optional[TransformerEncoderCFG]
            Stack configuration for the transformer decoder, built with
            :meth:`TransformerEncoderCFG.create` (``num_layers``, ...).
            Required when ``decoder_type == "transformer"``.
        use_conv_pos: bool
            Ablation switch for convolutional relative positional embeddings.
            When enabled, a single shared conv positional encoder is added to
            the encoder inputs of BOTH the student and the teacher, followed
            by a LayerNorm over the sum. The module is intentionally NOT
            EMA-tracked (matching data2vec 2.0 with ``ema_encoder_only=True``,
            where the modality encoder stays outside the EMA'd blocks and the
            teacher reads the live student weights).
        conv_pos_style: str
            ``"d2v2"``: data2vec 2.0 stacked encoder — ``depth`` blocks of
            ``Conv1d(k=max(3, width//depth), groups)`` -> SamePad ->
            non-affine LayerNorm -> GELU, no weight norm; masked positions
            zero-filled (no count renormalization).
            ``"wav2vec2"``: single wide weight-normed conv
            (:class:`NormalizedMaskedConvPositionalEmbedding`) with
            count-renormalized masking.
        conv_pos_width: int
            Total kernel budget. d2v2 default 95 (=> per-layer k=19 at
            depth 5); wav2vec2-style kernel size (their default 128).
        conv_pos_depth: int
            Stack depth for the d2v2 style (ignored for wav2vec2).
        conv_pos_groups: int
            Group count of the conv(s).
        conv_pos_pre_ln: bool
            d2v2 ``conv_pos_pre_ln``: LayerNorm on the encoder input before
            the positional stack (ignored for wav2vec2).
    """
    teacher_encoder: nn.Module
    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg : TransformerLayerCFG,
        transformer_encoder_cfg : TransformerEncoderCFG,
        conv_decoder_cfg : D2vDecoderConfig,
        loss_fn: nn.Module = nn.MSELoss(reduction='none'),
        lr: float = 0.0004,
        adam_betas: tuple[float, float] = (0.9, 0.98),
        adam_eps: float = 1e-06,
        adam_weight_decay: float = 0.04,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.99999,
        ema_anneal_end_step: int = 100000,
        warmup_steps: int = 100000,
        average_top_k_layers: int = 8,
        resample_sr : int = 16000,
        original_sr : int = 16000,
        use_gradient_checkpointing: bool = False,
        compile_modules : bool = False,
        size : str = "base",
        use_packing : bool = False,
        use_ctx_supervision : bool = False,
        decoder_type: str = "conv",                                            # NEW: "conv" | "transformer"
        transformer_decoder_layers_cfg : Optional[TransformerLayerCFG] = None, # NEW
        transformer_decoder_cfg : Optional[TransformerEncoderCFG] = None,      # NEW
        use_conv_pos: bool = False,                       # NEW
        conv_pos_style: str = "d2v2",                     # NEW: "d2v2" | "wav2vec2"
        conv_pos_width: int = 95,                         # NEW: total kernel budget (d2v2 default 95; wav2vec2 kernel 128)
        conv_pos_depth: int = 5,                          # NEW: d2v2 stack depth (ignored for wav2vec2)
        conv_pos_groups: int = 16,                        # NEW
        conv_pos_pre_ln: bool = False,                    # NEW: d2v2 conv_pos_pre_ln (ignored for wav2vec2)
        **kwargs : dict[str, Any],
    ):

        super().__init__()
        self.sr = resample_sr
        self.use_packing = use_packing
        print(f"Use Packing: {self.use_packing}")
        self.original_sr = original_sr
        self.ema_end_step = ema_anneal_end_step
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.save_hyperparameters(
            ignore=["feature_encoder", "feature_extractor", "loss_fn"]
        )
        self.use_ctx_supervision = use_ctx_supervision
        if self.use_ctx_supervision:
            print("Using CTX supervision")

        assert decoder_type in ("conv", "transformer"), \
            f"decoder_type must be 'conv' or 'transformer', got {decoder_type}"
        self.decoder_type = decoder_type
        self.use_conv_pos = use_conv_pos

        self.extract_audio = feature_extractor
        self.audio_feature_norms : nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)

        self.loss_fn = loss_fn
        self.warmup_steps=warmup_steps

        # If size is large, then alter the encoder parameters to mimic VIT-Large. Should results in ~300m parameters.
        if size == "large":
            transformer_encoder_layers_cfg["nhead"] = 16
            transformer_encoder_layers_cfg["d_model"] = 1024
            transformer_encoder_layers_cfg["dim_feedforward"] = 1024 * 4
            transformer_encoder_cfg["num_layers"] = 24


        self.n_encoder_heads = transformer_encoder_layers_cfg["nhead"]
        self.encoder_embedding_dim = transformer_encoder_layers_cfg["d_model"]

        self.encoder = TorchtuneEncoder(
                d_model = self.encoder_embedding_dim,
                dim_feedforward = self.encoder_embedding_dim * 4,
                norm_first = False,
                nhead = self.n_encoder_heads,
                num_layers = transformer_encoder_cfg["num_layers"],
                use_rope = True,
                max_seq_len=8192,
                attn_dropout = kwargs.get("attn_dropout", 0.0),
                activation_dropout = kwargs.get("activation_dropout", 0.0),
                hidden_dropout= kwargs.get("hidden_dropout", 0.0),
                layer_drop= kwargs.get("layer_drop", 0.0)
                )

        # ------------------------------------------------------------------
        # Ablatable decoder: original conv decoder OR a shallow RoPE
        # transformer over the (context + mask token) sequence.
        # ------------------------------------------------------------------
        if self.decoder_type == "transformer":
            assert transformer_decoder_layers_cfg is not None, \
                "decoder_type='transformer' requires transformer_decoder_layers_cfg"
            assert transformer_decoder_cfg is not None, \
                "decoder_type='transformer' requires transformer_decoder_cfg"

            decoder_d_model = transformer_decoder_layers_cfg["d_model"]
            decoder_dropout = transformer_decoder_layers_cfg.get("dropout", 0.0)
            print(f"Using transformer decoder (RoPE), "
                  f"layers={transformer_decoder_cfg['num_layers']}, "
                  f"d_model={decoder_d_model}, "
                  f"nhead={transformer_decoder_layers_cfg['nhead']}")

            self.decoder = TorchtuneEncoder(
                d_model=decoder_d_model,
                dim_feedforward=transformer_decoder_layers_cfg["dim_feedforward"],
                norm_first=transformer_decoder_layers_cfg.get("norm_first", False),
                nhead=transformer_decoder_layers_cfg["nhead"],
                num_layers=transformer_decoder_cfg["num_layers"],
                use_rope=True,
                max_seq_len=8192,
                attn_dropout=decoder_dropout,
                activation_dropout=decoder_dropout,
                hidden_dropout=decoder_dropout,
                layer_drop=0.0,
            )
            self.decoder_mask_fill = "mask_token"        # NEW
            self.decoder_mask_noise_std = 0.0            # NEW
            if decoder_d_model != self.encoder_embedding_dim:
                self.decoder_proj_in = nn.Linear(self.encoder_embedding_dim, decoder_d_model)
                self.decoder_proj_out = nn.Linear(decoder_d_model, self.encoder_embedding_dim)
            else:
                self.decoder_proj_in = nn.Identity()
                self.decoder_proj_out = nn.Identity()
        else:
            self.decoder = Decoder1d(conv_decoder_cfg, input_dim=self.encoder_embedding_dim)
            self.decoder_proj_in = nn.Identity()
            self.decoder_proj_out = nn.Identity()
            self.decoder_mask_fill = conv_decoder_cfg.mask_fill
            self.decoder_mask_noise_std = conv_decoder_cfg.mask_noise_std
            assert self.decoder_mask_fill in ("mask_token", "noise"), \
                f"mask_fill must be 'mask_token' or 'noise', got {self.decoder_mask_fill}"
            print(f"Conv decoder: kernel_dropout={conv_decoder_cfg.kernel_dropout}, "
                  f"mask_fill={self.decoder_mask_fill}"
                  + (f" (std={self.decoder_mask_noise_std})"
                     if self.decoder_mask_fill == "noise" else ""))

        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim)
        self.local_feature_norms : nn.Module = nn.LayerNorm(self.encoder_embedding_dim)

        if self.use_conv_pos:
            self.conv_pos_style = conv_pos_style
            if conv_pos_style == "d2v2":
                print(f"Using d2v2.0 conv positional encoder "
                      f"(width={conv_pos_width}, depth={conv_pos_depth}, "
                      f"groups={conv_pos_groups}, pre_ln={conv_pos_pre_ln}), not EMA tracked")
                self.conv_pos_embedding = D2v2ConvPositionalEncoder(
                    embed_dim=self.encoder_embedding_dim,
                    width=conv_pos_width,
                    depth=conv_pos_depth,
                    groups=conv_pos_groups,
                    pre_ln=conv_pos_pre_ln,
                )
            elif conv_pos_style == "wav2vec2":
                print(f"Using wav2vec2 conv positional embeddings "
                      f"(kernel={conv_pos_width}, groups={conv_pos_groups}), not EMA tracked")
                self.conv_pos_embedding = NormalizedMaskedConvPositionalEmbedding(
                    hidden_size=self.encoder_embedding_dim,
                    num_conv_pos_embeddings=conv_pos_width,
                    num_conv_pos_embedding_groups=conv_pos_groups,
                )
            else:
                raise ValueError(
                    f"conv_pos_style must be 'd2v2' or 'wav2vec2', got {conv_pos_style}"
                )
            self.conv_pos_norm = nn.LayerNorm(self.encoder_embedding_dim)
        else:
            self.conv_pos_embedding = None
            self.conv_pos_norm = None

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.encoder_embedding_dim, requires_grad=True)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)

        for name, module in self.named_children():
            if name == 'conv_pos_embedding':
                continue
            if name == 'decoder' and self.decoder_type == "conv":
                module.apply(self._decoder_init_weights)
            else:
                module.apply(self._init_weights)

        self._init_teacher()
        if compile_modules:
            self._compile_operations()
            self.collate_fn = torch.compile(collate_fn)
            self.resample = torch.compile(resample)
            self.masked_loss = torch.compile(self.masked_loss)
        else:
            self.collate_fn = collate_fn


    def _decoder_init_weights(self, m):
        fn = nn.LayerNorm
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, fn)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m : nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_teacher(self):
        # NOTE: only the encoder is copied. The conv positional embedding is a
        # sibling module and is deliberately shared (no teacher copy, no EMA).
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)


    def _get_ema_decay(self):
        if self.global_step >= self.ema_end_step:
            return self.hparams.ema_end_decay
        r = self.hparams.ema_end_decay - self.hparams.ema_decay
        pct_remaining = 1 - self.global_step / self.ema_end_step
        return self.hparams.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        # Only encoder params are EMA'd; conv_pos_embedding is excluded by
        # construction (it is not part of self.encoder / self.teacher_encoder).
        r = self._get_ema_decay()
        for student, teacher in zip(self.encoder.parameters(),
                                    self.teacher_encoder.parameters()):
            teacher.data.mul_(r).add_((1 - r) * student.detach().data)


    def _compile_operations(self):
        compile_kwargs = {"dynamic": True}
        self.encoder.compile(**compile_kwargs)     # in-place: state_dict keys unchanged
        self.decoder.compile(**compile_kwargs)
        if self.conv_pos_embedding is not None:
            self.conv_pos_embedding.compile(**compile_kwargs)
        self._forward_teacher = torch.compile(self._forward_teacher, **compile_kwargs)

    def configure_optimizers(self):
        #Got it from Data2Vec2.0 https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py
        #Line 264
        no_decay = [p for pn, p in self.named_parameters()
                    if p.requires_grad and (len(p.shape) == 1 or pn.endswith(".bias"))]
        decay    = [p for pn, p in self.named_parameters()
                    if p.requires_grad and not (len(p.shape) == 1 or pn.endswith(".bias"))]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.adam_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
            eps=self.hparams.adam_eps,
        )
        cosine_annealing = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.max_steps
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": cosine_annealing, "interval": "step"}}
    def _make_targets(self, layer_outputs: List[torch.Tensor], padding_mask: torch.Tensor):
        """
        Calculates Instance Norm ignoring padded tokens.
        padding_mask: (batch_size, seq_len) where True means padded.
        """
        stacked_outputs = torch.stack(layer_outputs)  # [num_layers, batch, seq_len, features]

        active_mask = (~padding_mask).unsqueeze(0).unsqueeze(-1)

        sum_outputs = (stacked_outputs * active_mask).sum(dim=2, keepdim=True)
        active_count = active_mask.sum(dim=2, keepdim=True).clamp(min=1)
        mean = sum_outputs / active_count

        variance = (((stacked_outputs - mean) * active_mask) ** 2).sum(dim=2, keepdim=True) / active_count
        std = torch.sqrt(variance + 1e-5)

        normalized = (stacked_outputs - mean) / std
        normalized = normalized * active_mask

        y = normalized.mean(dim=0)                     #[batch, seq_len, features]
        return y


    def pack_context(self, x, ctx_mask, pad_multiple: int = 64):
        """
        x:        (B, S, E) local features
        ctx_mask: (B, S) True = NOT context (your convention)
        returns packed tokens, their original positions, and a pad mask
        """
        B, S, E = x.shape
        is_ctx = ~ctx_mask                                   # True = context
        n_ctx = is_ctx.sum(-1)                               # (B,)
        L = int(n_ctx.max().item())
        L = min(S, ((L + pad_multiple - 1) // pad_multiple) * pad_multiple)

        # stable sort: context positions first, original order preserved
        order = torch.argsort(~is_ctx, dim=-1, stable=True)  # (B, S)
        idx = order[:, :L]                                   # (B, L) original positions
        pad = torch.arange(L, device=x.device)[None] >= n_ctx[:, None]  # (B, L) True = pad

        x_ctx = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, E))
        x_ctx = x_ctx.masked_fill(pad.unsqueeze(-1), 0.0)    # see warning below
        return x_ctx, idx, pad

    def _apply_conv_pos(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Adds masked, renormalized conv positional embeddings, then applies a
        LayerNorm over the sum — matching fairseq's post-norm convention
        (``x = LayerNorm(x + pos_conv(x))`` before the first encoder layer).

        x:          (B, S, E)
        valid_mask: (B, S) True = token participates (context for the student,
                    non-padded for the teacher). Masked-out tokens contribute
                    nothing to the conv and receive no positional signal,
                    preventing target leakage into the student context.
        """
        return self.conv_pos_norm(x + self.conv_pos_embedding(x, valid_mask))

    @torch.no_grad()
    def _forward_teacher(self, x : torch.Tensor, padding_mask : torch.Tensor) -> torch.Tensor:
        # Teacher gets the SAME (student) conv positional embedding weights,
        # applied over all non-padded tokens. Runs under no_grad, so no
        # gradients flow into conv_pos_embedding from the teacher path.
        if self.use_conv_pos:
            x = self._apply_conv_pos(x, ~padding_mask)

        layer_outputs = []
        valid_tokens = ~padding_mask
        mask = valid_tokens.view(x.shape[0], 1, x.shape[1]).expand(x.shape[0], x.shape[1], x.shape[1])
        for i, layer in enumerate(self.teacher_encoder.layers):
            if self.teacher_encoder.norm_first: # Pre-Norm
                normed_x = layer['norm_sa'](x)
                x = x + layer['attn'](normed_x, normed_x, mask=mask)
                ffn_out = layer['mlp'](layer['norm_mlp'](x))
                x = x + ffn_out
                target_repr = ffn_out
            else:
                x = layer['norm_sa'](x + layer['attn'](x, x, mask=mask))
                ffn_out = layer['mlp'](x)
                x = layer['norm_mlp'](x + ffn_out)
                target_repr = ffn_out

            if (
                len(self.teacher_encoder.layers) - i
                <= self.hparams.average_top_k_layers
            ):
                layer_outputs.append(target_repr)

        if self.hparams.average_top_k_layers > 1:
            targets = self._make_targets(layer_outputs, padding_mask=padding_mask)
        else:
            targets = layer_outputs[-1]
        return targets

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Runs on GPU. Splits batch by SR, resamples, recombines.
        """
        assert "audio" in batch
        assert "ctx_mask" in batch
        assert "tgt_mask" in batch
        assert "ctx_tgt_mask" in batch
        assert "padding_mask" in batch
        assert "teacher_padding_mask" in batch

        # Initialize clean_scene unconditionally
        clean_scene = batch["audio"]
        # Add channel dimension to the final audio as well.
        if clean_scene.ndim != 3:
            clean_scene = clean_scene.unsqueeze(1)

        assert clean_scene.shape[1] == 1, f"Generated scene has more channels than in channels, 1"
        assert clean_scene.ndim == 3

        if self.sr != self.original_sr:
            clean_scene = resample(clean_scene, resample_sr=self.sr, original_sr=self.original_sr)
            new_length = clean_scene.shape[-1]
            #This is for the audio representations.
            #Teacher padding mask is already calculated over the 16khz.
            padding_mask_old = batch["padding_mask"]
            batch["padding_mask"] = F.interpolate(
                padding_mask_old.unsqueeze(1).float(),
                size=new_length,
                mode='nearest'
            ).squeeze(1).to(padding_mask_old.dtype)

        #This padding mask has True for real audio and False for padding.
        clean_scene = masked_instance_normalize(clean_scene, batch["padding_mask"])

        # Cast to bfloat16 and flatten batch and samples dimensions
        clean_scene = clean_scene.to(torch.bfloat16)

        return (clean_scene,
                batch["ctx_mask"],
                batch["tgt_mask"],
                batch["ctx_tgt_mask"],
                batch["teacher_padding_mask"])


    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation for stability
            self._step_teacher()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        audio, ctx_masks, target_indices, ctx_and_target_masks, teacher_padding_mask = batch
        out = self(audio,
                   ctx_masks, target_indices, ctx_and_target_masks,
                   teacher_padding_mask,
                   use_packed=self.use_packing)

        target_variance = self.compute_var(out["targets"])
        pred_variance = self.compute_var(out["preds"])

        log_data = {
            "train/loss": out["loss"],
            "ema": self._get_ema_decay(),
            "train/target_variance": target_variance,
            "train/pred_variance": pred_variance,
        }

        self.log_dict(log_data, prog_bar=True, sync_dist=True)
        return out

    def masked_loss(self, pred, target, target_indices):
        """
        Calculates the masked loss using broadcasting to avoid memory-heavy repeats.

        pred:   Tensor of shape [(B * N), T, D]
        target: Tensor of shape [B, T, D]
        mask:   Tensor of shape [B, N, T]
        """
        B, N, _ = target_indices.shape
        D = pred.shape[-1]

        pred_reshaped = pred.view(B, N, -1, D)           # (B, N, T, D)
        target = target.unsqueeze(1).expand(B, N, -1, D)         # (B, N, T, D) — view, no copy

        loss = self.loss_fn(pred_reshaped, target)        # (B, N, T, D)
        loss_per_timestep = loss.mean(dim=-1)             # (B, N, T)
        masked = loss_per_timestep * target_indices       # (B, N, T)
        return masked.sum() / (target_indices.sum() + 1e-8)


    def _extract_audio(self,
                        audio: torch.Tensor):

        local_features = self.extract_audio(audio)
        local_features = self.audio_feature_norms(local_features)
        local_features = self.post_extraction_mapper(local_features)
        local_features = self.local_feature_norms(local_features)
        return local_features


    def forward(self, audio, ctx_masks, target_indices, ctx_and_target_masks,
                teacher_padding_masks, use_packed: bool = False) -> ForwardReturn:
        local_features = self._extract_audio(audio)
        # Teacher applies its own conv pos internally (same shared weights,
        # masked with the teacher padding mask) — see _forward_teacher.
        targets = self._forward_teacher(local_features.detach(),
                                        padding_mask=teacher_padding_masks)

        if self.use_conv_pos:
            student_features = self._apply_conv_pos(local_features, ~ctx_masks)
        else:
            student_features = local_features

        if use_packed:
            contextual_features = self.encoder_forward_packed(student_features, ctx_masks)
        else:
            contextual_features = self.encoder_forward(student_features, src_key_padding_mask=ctx_masks)
        preds = self.decoder_forward(contextual_features, ctx_masks,
                                    padding_mask=teacher_padding_masks,
                                    nr_targets=target_indices.shape[1],
                                    src_key_padding_mask=ctx_and_target_masks)
        loss = self.masked_loss(preds, targets, target_indices)
        return ForwardReturn(local_features=local_features,
                            contextual_features=contextual_features,
                            loss=loss, preds=preds, targets=targets)

    def decoder_forward(self, contextual_features, ctx_mask, padding_mask,
                        nr_targets, src_key_padding_mask=None):
        B, seq_len, E = contextual_features.shape

        if self.decoder_mask_fill == "noise":
            # NEW: d2v2-style fill — fresh N(0, std) noise instead of a learned token.
            # No grad, sampled per forward, shared across the N target groups.
            tgt = contextual_features.new_empty(B, seq_len, E).normal_(
                0.0, self.decoder_mask_noise_std)
        else:
            tgt = self.mask_token.expand(B, seq_len, E)                # (B, T, E)

        # Blend context in via masking (unchanged)
        ctx_mask_f = (~ctx_mask).unsqueeze(-1).to(contextual_features.dtype)
        tgt = tgt * ctx_mask.unsqueeze(-1).to(tgt.dtype) + contextual_features * ctx_mask_f

        tgt = repeat(tgt, 'B S E -> (B N) S E', N=nr_targets)

        src_key_padding_mask = rearrange(src_key_padding_mask, 'B N S -> (B N) S')
        padding_mask = repeat(padding_mask, 'B S -> (B N) S', N=nr_targets)
        is_context_or_tgt = (~src_key_padding_mask) & (~padding_mask)

        if self.decoder_type == "transformer":
            # RoPE transformer decoder: tokens sit at their original
            # positions (no packing here), so default positions are correct.
            # TorchtuneEncoder expects True = ignore in src_key_padding_mask.
            tgt = self.decoder_proj_in(tgt)
            tgt = self.decoder(tgt, src_key_padding_mask=~is_context_or_tgt)
            tgt = self.decoder_proj_out(tgt)
        else:
            tgt = self.decoder(tgt, is_context_or_tgt)
        return tgt

    def encoder_forward(self,
    x_contexts: torch.Tensor,
    src_key_padding_mask : torch.BoolTensor | None = None
    ) -> torch.Tensor:

        contextual_features = self.encoder(x_contexts,
                                            src_key_padding_mask = src_key_padding_mask)

        return contextual_features

    def encoder_forward_packed(self, local_features, ctx_mask):
        B, S, E = local_features.shape
        x_ctx, idx, pad = self.pack_context(local_features, ctx_mask)

        # Declare varying dims before the compiled call: batch (0) and
        # packed length (1). Harmless no-ops in eager mode.
        torch._dynamo.mark_dynamic(x_ctx, 0)
        torch._dynamo.mark_dynamic(x_ctx, 1)
        torch._dynamo.mark_dynamic(idx, 0)
        torch._dynamo.mark_dynamic(idx, 1)
        torch._dynamo.mark_dynamic(pad, 0)
        torch._dynamo.mark_dynamic(pad, 1)

        out = self.encoder(x_ctx, src_key_padding_mask=pad, input_pos=idx)

        # Scatter back eagerly — cheap, and scatter_ on a cloned expand is
        # exactly the kind of in-place-on-view pattern best kept out of graphs.
        out = torch.where(pad.unsqueeze(-1), self.mask_token.to(out.dtype), out)
        full = self.mask_token.expand(B, S, E).to(out.dtype).clone()
        full.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, E), out)
        return full
    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            # Use y.device to prevent cross-device DDP crashes!
            zc = torch.tensor(y.size(0), device=y.device, dtype=y.dtype)
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    @torch.inference_mode()
    def get_audio_representation(self, audio : torch.Tensor, attention_padding_mask : torch.tensor = None):
        local_features = self._extract_audio(audio)
        if self.use_conv_pos:
            if attention_padding_mask is not None:
                valid = ~attention_padding_mask
            else:
                valid = torch.ones(
                    local_features.shape[:2],
                    dtype=torch.bool,
                    device=local_features.device,
                )
            local_features = self._apply_conv_pos(local_features, valid)
        contextual_features = self.encoder_forward(local_features,
                                                   src_key_padding_mask = attention_padding_mask)
        return contextual_features