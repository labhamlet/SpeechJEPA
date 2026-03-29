import torch
from ax.service.managed_loop import optimize
from torchmetrics.text import WordErrorRate
from torchaudio.models.decoder import ctc_decoder

def optimize_decoding_hyperparameters(model, dev_dataloader, device="cuda"):
    model.eval()
    model.to(device)
    
    cached_emissions = []
    cached_lengths = []
    cached_targets =[]
    
    # 1. Run forward pass ONCE
    with torch.no_grad():
        for batch in dev_dataloader:
            audio = batch["audio"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            
            # Get logits
            logits = model(audio, padding_mask)
            emissions = torch.nn.functional.log_softmax(logits, dim=-1).cpu().contiguous()
            
            # Get lengths
            raw_audio_lengths = padding_mask.sum(dim=-1)
            feat_lengths = model.model._get_feat_extract_output_lengths(raw_audio_lengths)
            
            cached_emissions.append(emissions)
            cached_lengths.append(feat_lengths.cpu().to(torch.int32))
            cached_targets.extend(batch["text"])
    
    # The evaluation function Ax will call for each trial
    def evaluate_trial(parameters):
        alpha = parameters.get("alpha")
        beta = parameters.get("beta")
        
        # Instantiate decoder with the proposed hyperparameters
        decoder = ctc_decoder(
            lexicon=model.decoder_files.lexicon,
            tokens=[w.lower() for w in model.labels],
            lm=model.decoder_files.lm,
            nbest=1,
            beam_size=50,             # Keep beam size large and fixed
            lm_weight=alpha,           # Proposed by Ax
            word_score=beta,           # Proposed by Ax
            blank_token="-",
            sil_token="|"
        )
        
        wer_metric = WordErrorRate()
        all_preds =[]
        
        # Decode the cached emissions
        for emissions, lengths in zip(cached_emissions, cached_lengths):
            beam_results = decoder(emissions, lengths)
            preds =[" ".join(res[0].words).strip().upper() for res in beam_results]
            all_preds.extend(preds)
            
        # Return the Objective metric (WER) as a float
        return wer_metric(all_preds, cached_targets).item()

    print("Running Ax Bayesian Optimization")
    
    # Let Ax run 30-50 trials to find the global minimum WER
    best_parameters, values, experiment, ax_model = optimize(
        parameters=[
            {"name": "alpha", "type": "range", "bounds": [0.0, 5.0]}, # LM Weight
            {"name": "beta", "type": "range", "bounds": [-5.0, 5.0]}, # Word Score
        ],
        evaluation_function=evaluate_trial,
        objective_name="wer",
        minimize=True,   # We want to minimize WER
        total_trials=30  # 30 is usually plenty for a 2D space
    )
    
    print(f"Optimization Complete! Best params: {best_parameters}")
    print(f"Best Dev WER: {values[0]['wer'] * 100:.2f}%")
    
    return best_parameters