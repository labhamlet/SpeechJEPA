import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from transformers import AutoModel, AutoConfig
from .utils import get_tri_stage_schedule
import numpy as np 


class HuggingFaceASRForCTC(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        bundle,
        lr: float = 1e-4,    
        total_steps: int = 80000,
        freeze_encoder_updates: int = 10000, 
        mask_time_prob: float = 0.75,      # was 0.075
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,      # added
        mask_feature_prob: float = 0.256,    # was 0.004
        mask_feature_length: int = 64,
        mask_feature_min_masks: int = 0,   # added
        layer_drop : int = 0.1, 
        activation_dropout: int = 0.1,
        hidden_dropout: int = 0.0,
        attention_dropout: int = 0.0,
        lm_dropout: int = 0.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['bundle'])

        self.bundle = bundle 
        self.labels = self.bundle.get_labels()   
        try:
            self.pad_token_id = self.labels.index("-")
        except ValueError:
            print("Warning: '-' not found in labels! Defaulting pad_token_id to 0.")
            self.pad_token_id = 0        
    
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            mask_time_prob=mask_time_prob, 
            mask_time_length=mask_time_length, 
            mask_time_min_masks=mask_time_min_masks,
            mask_feature_prob=mask_feature_prob, 
            mask_feature_length=mask_feature_length,
            mask_feature_min_masks=mask_feature_min_masks,
            apply_spec_augment=True,
            layerdrop=layer_drop, 
            activation_dropout=activation_dropout,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
        )
        self.dropout = nn.Dropout(lm_dropout)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        
        # We always want the CNN feature extractor frozen during ASR fine-tuning
        if hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()
            
        # Freeze the entire transformer up to `freeze_encoder_updates`
        if self.hparams.freeze_encoder_updates > 0:
            self.model.requires_grad_(False)

        # Custom Classification CTC Head
        self.lm_head = nn.Linear(config.hidden_size, len(self.labels))
        
        # Separate metrics
        self.train_wer_metric = WordErrorRate()
        self.val_wer_metric = WordErrorRate()
        self.test_wer_metric = WordErrorRate()

        self.beam_search_dev = None
        self.beam_search_test = None

        self.model.training = True

    def prepare_data(self):
        download_pretrained_files("librispeech-4-gram")

    def setup(self, stage=None):
        self.decoder_files = download_pretrained_files("librispeech-4-gram")
        self.beam_search_dev = self._setup_torchaudio_decoder(beam_size=50)
        self.beam_search_test = self._setup_torchaudio_decoder(beam_size=50)

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.hparams.freeze_encoder_updates:
            self.print(f"Unfreezing encoder at step {self.global_step}!")
            self.model.requires_grad_(True)
            # Make sure the CNN feature extractor remains frozen even after unfreezing
            if hasattr(self.model, "freeze_feature_encoder"):
                self.model.freeze_feature_encoder()
            self.model.training=True

    def _setup_torchaudio_decoder(self, beam_size, lm_weight=2.0, word_score=-1.0):
        return ctc_decoder(
            lexicon=self.decoder_files.lexicon,
            tokens=[w.lower() for w in self.labels],
            lm=self.decoder_files.lm,
            nbest=1,
            beam_size=beam_size, 
            lm_weight=lm_weight,        
            word_score=word_score,      
            blank_token="-",      
            sil_token="|"         
        )

    def forward(self, audio, padding_mask):
        if audio.ndim == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)

        model_kwargs = {}

        feat_extract_norm = getattr(self.model.config, "feat_extract_norm", "layer")
        if feat_extract_norm == "layer":
            model_kwargs["attention_mask"] = padding_mask.long()

        outputs = self.model(audio, **model_kwargs)
        x = outputs.last_hidden_state
        return self.lm_head(self.dropout(x))

    def _greedy_decode(self, logits, lengths):
        preds = torch.argmax(logits, dim=-1) # (B, T)
        predictions =[]
        for i in range(preds.size(0)):
            pred = preds[i][:lengths[i]]
            pred = torch.unique_consecutive(pred)
            pred = pred[pred != self.pad_token_id]
            
            raw_text = "".join([self.labels[p] for p in pred]).replace("|", " ")
            clean_text = " ".join(raw_text.split())
            
            predictions.append(clean_text)
        return predictions

    def training_step(self, batch, batch_idx):
        self.model.training = True
        audio, labels, padding_mask, text_targets = (
            batch["audio"], batch['labels'], batch["padding_mask"], batch["text"]
        )
        
        logits = self(audio, padding_mask)
        
        loss = None
        if labels is not None:
            raw_audio_lengths = padding_mask.sum(-1)
            input_lengths = self.model._get_feat_extract_output_lengths(raw_audio_lengths).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs, flattened_targets, input_lengths, target_lengths,
                    blank=self.pad_token_id, reduction="sum", zero_infinity=True,
                )
                loss = loss / audio.size(0)     # normalize by #sentences ≈ sentence_avg=True
            with torch.no_grad():
                preds = self._greedy_decode(logits.detach().cpu(), input_lengths.cpu())
                wer = self.train_wer_metric(preds, text_targets)

            self.log("train/ctc_loss", loss, prog_bar=True)
            self.log("train/wer_greedy", wer, prog_bar=True, on_step=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
            self.model.training = False
            audio, padding_mask, text_targets = batch["audio"], batch["padding_mask"], batch["text"]

            logits = self(audio, padding_mask)

            raw_audio_lengths = padding_mask.sum(dim=-1)
            feat_lengths = self.model._get_feat_extract_output_lengths(raw_audio_lengths).to(torch.long)

            predictions = self._greedy_decode(logits.detach().cpu(), feat_lengths.cpu())

            if batch_idx == 0:
                print(f"\nPredictions (greedy): {predictions[:2]}")
                print(f"Targets:              {text_targets[:2]}")

            self.val_wer_metric.update(predictions, text_targets)

    def on_validation_epoch_end(self):
        val_wer = self.val_wer_metric.compute()
        self.log("val/wer_greedy", val_wer, sync_dist=True)
        self.print(f"\n---> [Step {self.global_step}] Validation WER (greedy): {val_wer * 100:.2f}% <---")
        self.val_wer_metric.reset()

    def test_step(self, batch, batch_idx):
            self.model.training = False
            audio, padding_mask, text_targets = batch["audio"], batch["padding_mask"], batch["text"]

            logits = self(audio, padding_mask)

            raw_audio_lengths = padding_mask.sum(dim=-1)
            feat_lengths = self.model._get_feat_extract_output_lengths(raw_audio_lengths).to(torch.long)

            predictions = self._greedy_decode(logits.detach().cpu(), feat_lengths.cpu())

            if batch_idx == 0:
                print(f"\nPredictions (greedy): {predictions[:2]}")
                print(f"Targets:              {text_targets[:2]}")

            self.test_wer_metric.update(predictions, text_targets)

    def on_test_epoch_end(self):
        test_wer = self.test_wer_metric.compute()
        self.log("test/wer_greedy", test_wer, sync_dist=True)
        self.print(f"\nFinal Test WER (greedy): {test_wer * 100:.2f}%")
        self.test_wer_metric.reset()

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,           # 5e-5
                betas=(0.9, 0.98),            # adam_betas: (0.9,0.98)
                eps=1e-8                      # adam_eps: 1e-08
            )
            scheduler = get_tri_stage_schedule(optimizer, 
                                               final_lr_scale=0.05,
                                               total_steps=self.hparams.total_steps)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}