import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from utils import get_identity_from_cfg
from data_modules import SSLDataModule, NoisySSLDataModule

from wavjepa.jepa_quantized import JEPAQuantized
from wavjepa.jepa_d2v2 import JEPA

from wavjepa.masking import SpeechMasker
from wavjepa.extractors import ConvFeatureExtractor, Extractor
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

# Component registries
NETWORKS = {"JEPA": JEPA,
            "JEPAQuantized": JEPAQuantized}
MASKERS = {
    "speech-masker": SpeechMasker
}
EXTRACTORS = {
    "wav2vec2": ConvFeatureExtractor,
}
ENCODERS = {
    "Transformer": {
        "LayerCFG": TransformerLayerCFG,
        "EncoderCFG": TransformerEncoderCFG,
    }
}

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Disable cuDNN benchmarking because we do not have consistent input sizes.
# Enable torch.compile with dynamic shapes later!
# Also check the convolutional positonal encodings.
torch.backends.cudnn.benchmark = False


class ComponentFactory:
    """Factory class for creating model components with proper error handling."""
    
    @staticmethod
    def create_extractor(cfg) -> Extractor:
        """Create and configure the extractor component."""
        extractor_class = EXTRACTORS.get(cfg.extractor.name)
        if extractor_class is None:
            raise ValueError(
                f"Unknown extractor type: {cfg.extractor.name}. "
                f"Available extractors: {list(EXTRACTORS.keys())}"
            )
        
        weight_sharing = cfg.extractor.get("share_weights_over_channels", None)
        return extractor_class(
                conv_layers_spec=eval(cfg.extractor.conv_layers_spec),
                in_channels=cfg.data.in_channels,
                depthwise = cfg.extractor.depthwise,
                share_weights_over_channels = weight_sharing,
            )
    
    
    @staticmethod
    def create_masker(cfg):
        """Create and configure the masker component."""
        masker_class = MASKERS.get(cfg.masker.name)
        if masker_class is None:
            raise ValueError(
                f"Unknown masker type: {cfg.masker.name}. "
                f"Available maskers: {list(MASKERS.keys())}"
            )
        
        if cfg.masker.name == "speech-masker":
            return SpeechMasker(
                    target_masks_per_context=cfg.masker.target_masks_per_context,
                    target_prob=cfg.masker.target_prob,
                    target_length=cfg.masker.target_length,
                    ratio_cutoff=cfg.masker.ratio_cutoff,
                    channel_based_masking=cfg.masker.channel_based_masking,
                    min_context_len = cfg.masker.min_context_len,
                )

    
    @staticmethod
    def create_network(cfg, extractor : Extractor) -> JEPA:
        """Create and configure the main network."""
        network_class = NETWORKS.get(cfg.model)
        if network_class is None:
            raise ValueError(
                f"Unknown network type: {cfg.model}. "
                f"Available networks: {list(NETWORKS.keys())}"
            )
        
        try:
            return network_class(
                feature_extractor=extractor,
                transformer_encoder_cfg = TransformerEncoderCFG.create(), 
                transformer_encoder_layers_cfg = TransformerLayerCFG.create(),
                lr=cfg.optimizer.lr,
                ema_decay=cfg.trainer.ema_decay,
                ema_end_decay=cfg.trainer.ema_end_decay,
                ema_anneal_end_step=cfg.trainer.ema_anneal_end_step,
                adam_betas=(cfg.optimizer.b1, cfg.optimizer.b2),
                adam_weight_decay=cfg.optimizer.weight_decay,
                resample_sr=cfg.data.sr,
                original_sr=cfg.data.original_sr,
                compile_modules = cfg.trainer.compile_modules,
                average_top_k_layers = cfg.trainer.average_top_k_layers,
                warmup_steps=cfg.trainer.warmup_steps,
                size = cfg.trainer.get("size", "base"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create network instance: {str(e)}")


def setup_logger(cfg) -> TensorBoardLogger:
    """Set up TensorBoard logger with proper configuration."""
    identity = get_identity_from_cfg(cfg)
    return TensorBoardLogger(
        f"{cfg.save_dir}/tb_logs_speech_jepa_asr",
        name=identity.replace("_", "/"),
    )


def setup_callbacks(cfg):
    """Set up training callbacks."""
    identity = get_identity_from_cfg(cfg)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.save_dir}/saved_models_speech_jepa_d2v2_like/{identity.replace('_', '/')}",
        filename="{step}",
        verbose=True,
        every_n_train_steps=10000,
        save_last=True,
        enable_version_counter=True,
        save_top_k=-1,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_callback, lr_monitor]


def setup_trainer(cfg, logger, callbacks) -> pl.Trainer:
    """Set up PyTorch Lightning trainer with proper configuration."""
    num_gpus = int(cfg.trainer.num_gpus)
    if num_gpus > 1:
        strategy = DDPStrategy(static_graph=False, find_unused_parameters=False)
    else:
        strategy = "auto"
    return pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.steps,
        precision=cfg.trainer.precision,
        deterministic=False,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=100,
        gradient_clip_algorithm="norm",
        gradient_clip_val=5,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=num_gpus,
        strategy=strategy
    )


def create_data_module(cfg) -> pl.LightningDataModule:
    """Create and configure the data module."""
    factory = ComponentFactory()
    masker = factory.create_masker(cfg)

    if "Noisy" in cfg.data.name:
        return NoisySSLDataModule(
            data_dir = cfg.data.data_dir,
            masker = masker, 
            noise_path = cfg.data.noise_path,
            rir_path = cfg.data.rir_path,
            noise_and_rir_sr = cfg.data.original_sr,
            data_sr=cfg.data.sr,
            min_sample_len = cfg.data.min_sample_len,
            max_sample_len = cfg.data.max_sample_len,
            target_batch_size = cfg.data.target_batch_size, 
            max_batch_size = cfg.data.max_batch_size,
            loudness_normalize = cfg.data.loudness_normalize,
            target_masks_per_context = cfg.masker.target_masks_per_context,
            conv_kernel = eval(cfg.extractor.conv_kernel),
            conv_stride = eval(cfg.extractor.conv_stride),
            bucket_limits = cfg.data.bucket_limits,
            pin_memory = True,
        )
    else:
        return SSLDataModule(
            data_dir = cfg.data.data_dir,
            masker = masker, 
            min_sample_len = cfg.data.min_sample_len,
            max_sample_len = cfg.data.max_sample_len,
            target_batch_size = cfg.data.target_batch_size, 
            max_batch_size = cfg.data.max_batch_size,
            loudness_normalize = cfg.data.loudness_normalize,
            conv_kernel = eval(cfg.extractor.conv_kernel),
            conv_stride = eval(cfg.extractor.conv_stride),
            target_masks_per_context = cfg.masker.target_masks_per_context,
            bucket_limits = cfg.data.bucket_limits,
            pin_memory = True,
        )

def build_model(cfg) -> torch.nn.Module:
    """Build the complete model with all components."""
    factory = ComponentFactory()
    
    # Create components in order of dependency
    extractor = factory.create_extractor(cfg)
    network = factory.create_network(cfg, extractor)

    return network



def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg):
    """Main training function."""
    try:
        seed_everything(cfg.seed, workers=True)
        
        # Setup training components
        logger = setup_logger(cfg)
        callbacks = setup_callbacks(cfg)
        trainer = setup_trainer(cfg, logger, callbacks)
        
        # Build model and data
        model = build_model(cfg)
        data_module = create_data_module(cfg)
        
        # Start training
        trainer.fit(model, data_module, ckpt_path = cfg.get("ckpt_path", None))

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    cleanup_memory()  # Clean up before starting
    main()
    cleanup_memory()  # Clean up after finishing