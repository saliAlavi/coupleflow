from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal
from .baseline_config import BaselineConfig

@dataclass
class EnvelopeTransformerConfig(BaselineConfig):
    """Configuration for EnvelopeTransformer model.
    Focuses on envelope inpainting using PyTorch's native transformer with causal masking.
    """
    
    # Available context durations and strategies
    CONTEXT_SECONDS: ClassVar[Tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    MASK_STRATEGIES: ClassVar[Tuple[str, ...]] = ('deterministic', 'random')
    
    # Model Architecture Configuration
    in_channel: int = 64            # Number of input EEG channels (not used but kept for compatibility)
    d_model: int = 128             # Model dimension
    n_head: int = 8               # Number of attention heads
    n_layers: int = 8             # Number of transformer layers
    d_ff: int = 2048             # Feed-forward network dimension
    max_seq_len: int = 640      # Maximum sequence length for positional encoding
    
    # Training Configuration
    dropout: float = 0.1          # Dropout rate
    warmup_epochs: int = 20       # Number of warmup epochs
    weight_decay: float = 0.0     # Weight decay for optimizer
    needs_transformer_schedule: bool = True
    
    # Contextual Configuration
    is_contextual: bool = True  # Flag for contextual models (e.g., inpainting models)
    context_mask_strategy: Literal['deterministic', 'random'] = "deterministic"  # Masking strategy
    context_seconds: float = 1.0  # Number of seconds to mask for context (ignored when strategy is random)
    mask_value: float = 0.0  # Value to mask the context with
    
    # Data Configuration
    train_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist/dev.csv"
    val_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist/dev.csv"
    test_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist/test.csv"
    data_base_dir: str = "/home/karan/sda_link/datasets/Dutch_Speech/mls_dutch_envelope_features"
    
    # Experiment Configuration
    experiment_name: str = "envelope_transformer"
    autoregressive: bool = True  # This model is always autoregressive due to causal masking
    checkpoint_dir: str = "checkpoints_ssl"
    wandb_project: str = "ssl_envelope"
    
    def __post_init__(self):
        """Validate configuration and set experiment name."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'in_channel': self.in_channel,
            'd_model': self.d_model,
            'n_head': self.n_head,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
        } 