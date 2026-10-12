from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal
from .baseline_config import BaselineConfig

@dataclass
class EnvelopeLSTMConfig(BaselineConfig):
    """Configuration for EnvelopeLSTM model.
    Focuses on envelope inpainting using LSTM layers with normalization.
    """
    
    # Available context durations and strategies
    CONTEXT_SECONDS: ClassVar[Tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    MASK_STRATEGIES: ClassVar[Tuple[str, ...]] = ('deterministic', 'random')
    
    # Model Architecture Configuration
    in_channel: int = 64            # Number of input EEG channels (not used but kept for compatibility)
    hidden_dim: int = 256          # Hidden dimension for LSTM
    n_layers: int = 4             # Number of LSTM layers
    bidirectional: bool = False   # Whether to use bidirectional LSTM
    
    # Training Configuration
    dropout: float = 0.1          # Dropout rate
    warmup_epochs: int = 20       # Number of warmup epochs
    weight_decay: float = 0.0     # Weight decay for optimizer
    needs_transformer_schedule: bool = False  # LSTM doesn't need transformer learning rate schedule
    
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
    experiment_name: str = "envelope_lstm"
    autoregressive: bool = True  # LSTM processes sequence autoregressively
    checkpoint_dir: str = "checkpoints_ssl"
    wandb_project: str = "ssl_envelope"
    
    def __post_init__(self):
        """Validate configuration and set experiment name."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'in_channel': self.in_channel,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
        } 