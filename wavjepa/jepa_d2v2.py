import copy
import transformers 
import torchaudio 

from typing import List, Any, Optional

import torch
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
import pytorch_lightning as pl

from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG
from data_modules.scene_module import generate_scenes_batch

from wavjepa.pos_embed import Wav2Vec2PositionalConvEmbedding, NormalizedMaskedConvPositionalEmbedding
from wavjepa.modules import TorchtuneEncoder, Decoder1d, D2vDecoderConfig


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
    """
    teacher_encoder: nn.Module
    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg : TransformerLayerCFG,
        transformer_encoder_cfg : TransformerEncoderCFG,
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
        clean_audio_ratio : float = 0.0,
        **kwargs : dict[str, Any],
    ):
        
        super().__init__(**kwargs)
        self.sr = resample_sr 
        self.original_sr = original_sr
        self.ema_end_step = ema_anneal_end_step
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.save_hyperparameters(
            ignore=["feature_encoder", "feature_extractor", "loss_fn"]
        )
        self.extract_audio = feature_extractor
        self.audio_feature_norms : nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)

        self.loss_fn = loss_fn
        self.clean_audio_ratio = clean_audio_ratio
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
                max_seq_len=8192
                )

        self.decoder = Decoder1d(D2vDecoderConfig, input_dim=self.encoder_embedding_dim)

        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim)
        self.local_feature_norms : nn.Module = nn.LayerNorm(self.encoder_embedding_dim)

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.encoder_embedding_dim, requires_grad=True)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)
 
        for name, module in self.named_children():
            if name == 'decoder':
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
        r = self._get_ema_decay()
        for student, teacher in zip(self.encoder.parameters(), 
                                    self.teacher_encoder.parameters()):
            teacher.data.mul_(r).add_((1 - r) * student.detach().data)


    def _compile_operations(self):
        compile_kwargs = {
            "dynamic": True
        }
        self.encoder = torch.compile(self.encoder, **compile_kwargs)
        self.decoder = torch.compile(self.decoder, **compile_kwargs)
        self.teacher_encoder = torch.compile(self.teacher_encoder, **compile_kwargs)


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


    @torch.no_grad()
    def _forward_teacher(self, x : torch.Tensor, padding_mask : torch.Tensor) -> torch.Tensor:
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
        generated_scene = generate_scenes_batch.generate_scene(
            source=batch["audio"], 
            noise=batch.get("noise", [None]),
            real_noise_length=batch.get("noise_length", None) ,
            noise_start_idx=batch.get("noise_start_idx", None),
            source_rir=batch.get("source_rir", [None]),
            noise_rirs=batch.get("noise_rirs", [None]),
            snr=batch.get("snr", None)
        )
        # Add channel dimension to the final audio as well.
        if clean_scene.ndim != 3:
            clean_scene = clean_scene.unsqueeze(1)
        if generated_scene.ndim != 3:
            generated_scene = generated_scene.unsqueeze(1)
        
        assert generated_scene.ndim == clean_scene.ndim
        assert generated_scene.shape[1] == 1, f"Generated scene has more channels than in channels, {generated_scene.shape}, 1"
        assert clean_scene.shape[1] == 1, f"Generated scene has more channels than in channels, {generated_scene.shape}, 1"
        assert clean_scene.ndim == 3

        if self.sr != self.original_sr:
            generated_scene = resample(generated_scene, resample_sr=self.sr, original_sr=self.original_sr)
            clean_scene = resample(clean_scene, resample_sr=self.sr, original_sr=self.original_sr)
            new_length = generated_scene.shape[-1]
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
        generated_scene = masked_instance_normalize(generated_scene, batch["padding_mask"])

        # Cast to bfloat16 and flatten batch and samples dimensions
        clean_scene = clean_scene.to(torch.bfloat16)
        generated_scene = generated_scene.to(torch.bfloat16)

        return (generated_scene, 
                clean_scene, 
                batch["ctx_mask"], 
                batch["tgt_mask"],
                batch["ctx_tgt_mask"],
                batch["teacher_padding_mask"])


    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation for stability
            self._step_teacher()
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        student_input, teacher_input, ctx_masks, target_indices, ctx_and_target_masks, teacher_padding_mask = batch
        out = self(student_input, teacher_input, ctx_masks, target_indices, ctx_and_target_masks, teacher_padding_mask)

        # Enhanced logging
        log_data = {
            "train/loss": out['loss'],
            "ema" : self._get_ema_decay(),
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


    def forward(self, 
                student_input : torch.Tensor, 
                teacher_input : torch.Tensor, 
                ctx_masks, 
                target_indices, 
                ctx_and_target_masks,
                teacher_padding_masks) -> ForwardReturn:
        
        # Teacher: Only has padding (no masked targets)
        teacher_features = self._extract_audio(teacher_input)
        teacher_features = teacher_features.detach()
        #Teacher only ignores the padding tokens
        targets = self._forward_teacher(teacher_features, 
                                        padding_mask=teacher_padding_masks)
        
        # Student: Has both targets (ctx_masks) AND padding
        local_features = self._extract_audio(student_input)
        #Here student ignroes the padding tokens + non context tokens                                     
        contextual_features = self.encoder_forward(local_features, src_key_padding_mask=ctx_masks)
        #Here decoder ignores the padding tokens + non target tokens.
        preds = self.decoder_forward(contextual_features, 
                                     ctx_masks, 
                                     padding_mask=teacher_padding_masks,
                                     nr_targets = target_indices.shape[1], 
                                     src_key_padding_mask=ctx_and_target_masks)
        

        loss = self.masked_loss(preds, targets, target_indices)
        

        return ForwardReturn(
            local_features=local_features,
            contextual_features=contextual_features,
            loss=loss,
            preds=preds,
            targets=targets,
        )

    def decoder_forward(self, 
                        contextual_features, 
                        ctx_mask, 
                        padding_mask, 
                        nr_targets, 
                        src_key_padding_mask=None):
        B, seq_len, E = contextual_features.shape

        # Start from all mask tokens
        tgt = self.mask_token.expand(B, seq_len, E)                # (B, T, E)

        # Blend context in via masking
        ctx_mask_f = (~ctx_mask).unsqueeze(-1).to(contextual_features.dtype)  # (B, T, 1)
        tgt = tgt * ctx_mask.unsqueeze(-1).to(tgt.dtype) + contextual_features * ctx_mask_f
        
        tgt = repeat(tgt, 'B S E -> (B N) S E', N=nr_targets)

        src_key_padding_mask = rearrange(src_key_padding_mask, 'B N S -> (B N) S')
        padding_mask = repeat(padding_mask, 'B S -> (B N) S', N=nr_targets)
        is_context_or_tgt = (~src_key_padding_mask) & (~padding_mask)  
        tgt = self.decoder(tgt, is_context_or_tgt)
        return tgt
    
    def encoder_forward(self, 
    x_contexts: torch.Tensor, 
    src_key_padding_mask : torch.BoolTensor | None = None
    ) -> torch.Tensor:

        contextual_features = self.encoder(x_contexts, 
                                            src_key_padding_mask = src_key_padding_mask)

        return contextual_features

    @torch.inference_mode()
    def get_audio_representation(self, audio : torch.Tensor, attention_padding_mask : torch.tensor = None):
        local_features = self._extract_audio(audio)
        contextual_features = self.encoder_forward(local_features, 
                                                   src_key_padding_mask = attention_padding_mask)
        return contextual_features