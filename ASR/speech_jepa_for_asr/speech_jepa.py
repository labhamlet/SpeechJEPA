import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from .utils import get_tri_state_schedule
import numpy as np 


def masked_instance_normalize(audio, mask):
    active_mask = mask.unsqueeze(1) 
    sum_audio = (audio * active_mask).sum(dim=-1, keepdim=True)
    active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = sum_audio / active_count
    
    variance = (((audio - mean) * active_mask) ** 2).sum(dim=-1, keepdim=True) / active_count
    std = torch.sqrt(variance)
    
    normalized = (audio - mean) / (std + 1e-5)
    return normalized * active_mask

def _compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: torch.LongTensor | None = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://huggingface.co/papers/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask



class SpeechJEPAForCTC(pl.LightningModule):
    def __init__(
        self,
        pretrained_jepa: pl.LightningModule,
        bundle,
        audio_token_func,
        lr: float = 1e-4,    
        total_steps: int = 80000,
        freeze_encoder_updates: int = 10000, 
        mask_time_prob: float = 0.065,
        mask_time_length: int = 10,
        mask_time_min_masks : int = 2,
        mask_feature_prob: float = 0.004,
        mask_feature_length: int = 64,
        mask_feature_min_masks : int = 0,
        dropout : float = 0.1,
        downsampling_factor: int = 320, 
        with_decoder: bool = False,
        use_superb: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_jepa'])

        self.bundle = bundle
        self.do_with_decoder = with_decoder
        self.use_superb = use_superb
        self.labels = self.bundle.get_labels()   
        try:
            self.pad_token_id = self.labels.index("-")
        except ValueError:
            print("Warning: '-' not found in labels! Defaulting pad_token_id to 0.")
            self.pad_token_id = 0        
        
        self.model = pretrained_jepa
        self.extract_audio = pretrained_jepa.extract_audio
        self.audio_feature_norms = pretrained_jepa.audio_feature_norms
        self.post_extraction_mapper = pretrained_jepa.post_extraction_mapper
        if hasattr(pretrained_jepa, "encoder_sin_cos_pos_embedding"):
            self.encoder_sin_cos_emb = pretrained_jepa.encoder_sin_cos_pos_embedding
        else:
            self.encoder_sin_cos_emb = None 
        
        if hasattr(pretrained_jepa, "decoder_sin_cos_pos_embedding"):
            self.decoder_sin_cos_emb = pretrained_jepa.decoder_sin_cos_pos_embedding
        else:
            self.decoder_sin_cos_emb = None 
        
        self.local_feature_norms = pretrained_jepa.local_feature_norms
        self.encoder = pretrained_jepa.encoder
        
        if with_decoder:
            self.decoder = pretrained_jepa.decoder 
            print("Performing ASR with the decoder")

    
        self._get_feat_extract_output_lengths = audio_token_func

        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length  
        self.mask_time_min_masks = mask_time_min_masks

        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length  
        self.mask_feature_min_masks = mask_feature_min_masks


        #Wav2Vec2.0 drops futures right before the transformer
        self.hidden_dropout = nn.Dropout(0.1)
        #CTC Dropout
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(pretrained_jepa.encoder_embedding_dim, len(self.labels))
        
        if self.mask_time_prob > 0.0 or self.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(pretrained_jepa.encoder_embedding_dim).uniform_())
        
        # Separate metrics to prevent training and eval steps from mixing states
        self.train_wer_metric = WordErrorRate()
        self.val_wer_metric = WordErrorRate()
        self.test_wer_metric = WordErrorRate()
        self.val_wer_metric_greedy = WordErrorRate()
        self.test_wer_metric_greedy = WordErrorRate()

        self.extract_audio.requires_grad_(False)    
        self.extract_audio.eval() 

        if self.hparams.freeze_encoder_updates > 0:
            self.audio_feature_norms.requires_grad_(False)
            self.audio_feature_norms.eval()
            self.encoder.requires_grad_(False)
            self.encoder.eval()
            self.local_feature_norms.requires_grad_(False)
            self.local_feature_norms.eval()
            self.post_extraction_mapper.requires_grad_(False)
            self.post_extraction_mapper.eval()
            if with_decoder:
                self.decoder.requires_grad_(False)
                self.decoder.eval()

        self.decoder_files = download_pretrained_files("librispeech-4-gram")
        self.beam_search_dev = self._setup_torchaudio_decoder(beam_size=50, lm_weight=2.0, word_score=-1.0)
        self.beam_search_test = self._setup_torchaudio_decoder(beam_size=50)
        if self.use_superb:
            self.layer_weights = nn.Parameter(
                torch.ones(pretrained_jepa.encoder.num_layers + 1)  # +1 for embedding layer
            )

    def on_train_batch_start(self, batch, batch_idx):
        self.extract_audio.eval()  # Always frozen, always eval
        if self.global_step < self.hparams.freeze_encoder_updates:
            self.audio_feature_norms.eval()
            self.encoder.eval()
            self.local_feature_norms.eval()
            self.post_extraction_mapper.eval()
            if self.do_with_decoder:
                self.decoder.eval()
        
        if self.global_step == self.hparams.freeze_encoder_updates:
            self.print(f"Unfreezing encoder at step {self.global_step}!")
            self.encoder.train()
            self.audio_feature_norms.train()
            self.local_feature_norms.train()
            self.post_extraction_mapper.train()
        
            self.encoder.requires_grad_(True)
            self.audio_feature_norms.requires_grad_(True)
            self.local_feature_norms.requires_grad_(True)
            self.post_extraction_mapper.requires_grad_(True)


    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://huggingface.co/papers/1904.08779).
        """


        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()
        if self.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.mask_time_prob,
                mask_length=self.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.mask_feature_prob,
                mask_length=self.mask_feature_length,
                min_masks=self.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states
    
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

    def forward(self, audio, attention_mask, padding_mask):
        audio = masked_instance_normalize(audio, padding_mask)

        with torch.no_grad():
            x = self.extract_audio(audio)

        x = self.audio_feature_norms(x)
        x = self.post_extraction_mapper(x)
        x = self.local_feature_norms(x)
        x = self._mask_hidden_states(x, attention_mask=~attention_mask)
        x = self.hidden_dropout(x)
        x = self.encoder(
            x, 
            src_key_padding_mask=attention_mask,
        )
        return self.lm_head(self.dropout(x))


    def _greedy_decode(self, logits, lengths):
        """Ultra-fast decoding to compute training WER without bogging down the GPU."""
        preds = torch.argmax(logits, dim=-1) # (B, T)
        predictions =[]
        for i in range(preds.size(0)):
            pred = preds[i][:lengths[i]]
            pred = torch.unique_consecutive(pred)
            pred = pred[pred != self.pad_token_id]
            
            # Replace boundaries with spaces
            raw_text = "".join([self.labels[p] for p in pred]).replace("|", " ")
            
            # Split and join cleans up any double/triple spaces
            clean_text = " ".join(raw_text.split())
            
            predictions.append(clean_text)
        return predictions

    def training_step(self, batch, batch_idx):
        audio, labels, padding_mask, attention_mask, text_targets = (
            batch["audio"], batch['labels'],
            batch["padding_mask"], batch["attention_mask"], batch["text"]
        )
        logits = self(audio, attention_mask, padding_mask)
        
        loss = None
        if labels is not None:
            # Padding mask is true where it is the "real" audio
            input_lengths = self._get_feat_extract_output_lengths(padding_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                                    log_probs, flattened_targets, input_lengths, target_lengths,
                                    blank=self.pad_token_id, reduction="mean",
                                    zero_infinity=True,    # was False
                                )

            with torch.no_grad():
                preds = self._greedy_decode(logits.detach().cpu(), input_lengths.cpu())
                wer = self.train_wer_metric(preds, text_targets)

            self.log("train/ctc_loss", loss, prog_bar=True)
            self.log("train/wer_greedy", wer, prog_bar=True, on_step=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        audio, padding_mask, attention_mask, text_targets = (
            batch["audio"], batch["padding_mask"], batch["attention_mask"], batch["text"]
        )
        
        logits = self(audio, attention_mask, padding_mask)
        # emissions = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).cpu().contiguous()
        
        raw_audio_lengths = padding_mask.sum(dim=-1)
        feat_lengths = self._get_feat_extract_output_lengths(raw_audio_lengths)
            
        # lengths_int32 = feat_lengths.cpu().to(torch.int32)
        # beam_results = self.beam_search_dev(emissions, lengths_int32)
        # predictions_lm = [" ".join(res[0].words).strip().upper() for res in beam_results]
        
        lengths_long = feat_lengths.cpu().to(torch.long)
        predictions_greedy = self._greedy_decode(logits.detach().cpu(), lengths_long)
        
        if batch_idx == 0:
            print(f"\nTargets:             {text_targets[:2]}")
            # print(f"Predictions (LM):    {predictions_lm[:2]}")
            print(f"Predictions (No LM): {predictions_greedy[:2]}")
            
        # self.val_wer_metric.update(predictions_lm, text_targets)
        self.val_wer_metric_greedy.update(predictions_greedy, text_targets)

    def on_validation_epoch_end(self):
        # val_wer_lm = self.val_wer_metric.compute()
        val_wer_greedy = self.val_wer_metric_greedy.compute()
        
        # self.log("val/wer_4gram", val_wer_lm, sync_dist=True)
        self.log("val/wer_greedy", val_wer_greedy, sync_dist=True)
        
        # self.print(f"\n---> [Step {self.global_step}] Validation WER | 4-gram LM: {val_wer_lm * 100:.2f}% | Greedy: {val_wer_greedy * 100:.2f}% <---")
        self.print(f"\n---> [Step {self.global_step}] Validation WER | Greedy: {val_wer_greedy * 100:.2f}% <---")

        # self.val_wer_metric.reset()
        self.val_wer_metric_greedy.reset()

    def test_step(self, batch, batch_idx):
        audio, padding_mask, attention_mask, text_targets = (
            batch["audio"], batch["padding_mask"], batch["attention_mask"], batch["text"]
        )
        
        logits = self(audio, attention_mask, padding_mask)
        # emissions = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).cpu().contiguous()
        
        raw_audio_lengths = padding_mask.sum(dim=-1)
        feat_lengths = self._get_feat_extract_output_lengths(raw_audio_lengths)
            
        # lengths_int32 = feat_lengths.cpu().to(torch.int32)
        # beam_results = self.beam_search_test(emissions, lengths_int32)
        # predictions_lm = [" ".join(res[0].words).strip().upper() for res in beam_results]

        lengths_long = feat_lengths.cpu().to(torch.long)
        predictions_greedy = self._greedy_decode(logits.detach().cpu(), lengths_long)
        
        if batch_idx == 0:
            print(f"\nTargets:             {text_targets[:2]}")
            # print(f"Predictions (LM):    {predictions_lm[:2]}")
            print(f"Predictions (No LM): {predictions_greedy[:2]}")
            
        # self.test_wer_metric.update(predictions_lm, text_targets)
        self.test_wer_metric_greedy.update(predictions_greedy, text_targets)

    def on_test_epoch_end(self):
        # test_wer_lm = self.test_wer_metric.compute()
        test_wer_greedy = self.test_wer_metric_greedy.compute()
        
        # self.log("test/wer_4gram", test_wer_lm, sync_dist=True)
        self.log("test/wer_greedy", test_wer_greedy, sync_dist=True)
        
        # self.print(f"\nFinal Test WER | 4-gram LM: {test_wer_lm * 100:.2f}% | Greedy (No LM): {test_wer_greedy * 100:.2f}%")
        self.print(f"\n---> [Step {self.global_step}] Test WER | Greedy: {test_wer_greedy * 100:.2f}% <---")

        self.test_wer_metric.reset()
        self.test_wer_metric_greedy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     betas= (0.9,0.98),
                                     lr=self.hparams.lr)
        scheduler = get_tri_state_schedule(optimizer, total_steps=self.hparams.total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
