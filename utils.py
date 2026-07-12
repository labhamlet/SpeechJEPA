def get_identity_from_cfg(cfg):
    identity = "Data={}_Rope={}_EMA={}_EMAEnd={}_EMASteps={}_".format(
        cfg.data.get("name", "Libri"),
        cfg.trainer.get("use_rope", True),
        cfg.trainer.get("ema_decay"),
        cfg.trainer.get("ema_end_decay"),
        cfg.trainer.get("ema_anneal_end_step"),
    )
 
    # --- conv positional embedding ablation (pos_embedding config group) ---
    use_conv_pos = cfg.pos_embedding.get("use_conv_pos", False)
    identity += f"ConvPos={use_conv_pos}_"
    if use_conv_pos:
        identity += "ConvPosStyle={}_ConvPosWidth={}_ConvPosDepth={}_ConvPosPreLN={}_".format(
            cfg.pos_embedding.get("style", "d2v2"),
            cfg.pos_embedding.get("width", 95),
            cfg.pos_embedding.get("depth", 5),
            cfg.pos_embedding.get("pre_ln", False),
        )
 
    # --- decoder ablation (decoder config group) ---
    decoder_type = cfg.decoder.get("name", "conv")
    identity += "Decoder={}_".format(decoder_type)
    if decoder_type == "transformer":
        identity += "DecoderLayers={}_DecoderDim={}_DecoderHeads={}_".format(
            cfg.decoder.transformer_decoder_cfg.get("num_layers"),
            cfg.decoder.transformer_decoder_layers_cfg.get("d_model"),
            cfg.decoder.transformer_decoder_layers_cfg.get("nhead"),
        )
    else:
        identity += "DecoderLayers={}_DecoderDim={}_".format(
            cfg.decoder.conv.get("nr_layers"),
            cfg.decoder.conv.get("embedding_dim"),
        )
        # --- kernel dropout ablation ---
        kernel_dropout = cfg.decoder.conv.get("kernel_dropout", True)
        mask_fill = cfg.decoder.conv.get("mask_fill", "mask_token")
        identity += f"KernelDropout={kernel_dropout}_MaskFill={mask_fill}_"
        if mask_fill == "noise":
            identity += "NoiseStd={}_".format(cfg.decoder.conv.get("mask_noise_std", 0.01))

    return identity
