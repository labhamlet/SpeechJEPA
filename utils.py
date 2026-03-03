def get_identity_from_cfg(cfg):
    identity = "Data={}_EMA={}_EMAEnd={}_EMASteps={}_".format(
        cfg.data.get("name", None),
        cfg.trainer.get("ema_decay"),
        cfg.trainer.get("ema_end_decay"),
        cfg.trainer.get("ema_anneal_end_step")
    )
    identity += "MaxBatchSize={}_NrGPUs={}_LR={}_LRWarmup={}_".format(
        cfg.data.get("max_batch_size"),
        cfg.trainer.get("num_gpus"),
        cfg.optimizer.get("lr"),
        cfg.trainer.get("warmup_steps")
    )
    identity += "TargetProb={}_TargetLen={}_MinContextBlock={}_ContextRatio={}".format(
        cfg.masker.get("target_prob"),
        cfg.masker.get("target_length"),
        cfg.masker.get("min_context_len"),
        cfg.masker.get("ratio_cutoff"),
    )
    return identity


def get_identity_from_cfg_denoise(cfg):
    identity = "Data={}_".format(
        cfg.data.get("name", None),
    )
    identity += "Extractor={}_InSeconds={}_".format(
        cfg.extractor.name,
        cfg.data.process_seconds,
    )
    identity += "BatchSize={}_NrSamples={}_NrGPUs={}_LR={}_".format(
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
        cfg.trainer.get("num_gpus"),
        cfg.optimizer.get("lr"),
    )
    identity += "Alpha={}".format(
        cfg.trainer.get("alpha", 0.0)
    )
    return identity
