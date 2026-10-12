from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal
from .baseline_config import BaselineConfig

@dataclass
class EnvelopeForecasterAutoregressiveConfig(BaselineConfig):
    """Configuration for EnvelopeForecasterAutoregressive model.
    Focuses on forecasting multiple future steps autoregressively based on 2 seconds of context.
    """
    
    # Model Architecture Configuration
    in_channel: int = 1            # Number of input channels (envelope is 1D)
    hidden_dim: int = 128          # Hidden dimension for RNN
    n_layers: int = 2              # Number of RNN layers
    rnn_type: Literal['lstm', 'gru', 'transformer'] = 'lstm'  # Type of model to use
    bidirectional: bool = False    # Whether to use bidirectional RNN (only for lstm/gru)
    nhead: int = 4                 # Number of attention heads (for transformer)
    
    # Training Configuration
    dropout: float = 0.1           # Dropout rate
    warmup_epochs: int = 10        # Number of warmup epochs
    weight_decay: float = 1e-5     # Weight decay for optimizer
    needs_transformer_schedule: bool = False  # Set to True for transformer models
    teacher_forcing_ratio: float = 0.5  # Ratio of teacher forcing during training
    
    # Forecasting Configuration
    context_seconds: float = 2.0   # Number of seconds of context (fixed at 2 seconds)
    prediction_seconds: float = 1.0  # Number of seconds to predict in each step
    future_seconds: float = 8.0    # Total number of seconds to predict into the future
    step_size: float = 0.5         # Step size in seconds for autoregressive prediction
    sampling_rate: int = 64        # Sampling rate in Hz
    
    # Contextual Configuration
    is_contextual: bool = True     # Flag for contextual models
    context_mask_strategy: Literal['deterministic', 'random'] = "deterministic"  # Masking strategy
    mask_value: float = 0.0        # Value to mask the context with
    
    # Data Configuration
    train_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist_10s/train.csv"
    val_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist_10s/dev.csv"
    test_filelist: str = "/home/karan/sda_link/GitHub/SSL_EEG_AAD/Dutch_Speech_SSL/filelist_10s/test.csv"
    data_base_dir: str = "/home/karan/sda_link/datasets/Dutch_Speech/mls_dutch_envelope_features"
    
    # Experiment Configuration
    experiment_name: str = "envelope_forecaster_autoregressive"
    autoregressive: bool = True    # RNN processes sequence autoregressively
    checkpoint_dir: str = "checkpoints_forecaster_autoregressive"
    wandb_project: str = "envelope_forecasting_autoregressive"
    
    # Loss Configuration
    loss_type: Literal['mse', 'mae', 'huber'] = 'mse'  # Type of loss function to use
    
    def __post_init__(self):
        """Validate configuration and set experiment name."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        
        # Set transformer schedule if using transformer
        if self.rnn_type == 'transformer':
            self.needs_transformer_schedule = True
            
        # Update experiment name with model details
        self.experiment_name = f"envelope_forecaster_autoregressive_{self.rnn_type}_{self.n_layers}layers_{self.hidden_dim}dim"
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'in_channel': self.in_channel,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'rnn_type': self.rnn_type,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'context_seconds': self.context_seconds,
            'prediction_seconds': self.prediction_seconds,
            'future_seconds': self.future_seconds,
            'step_size': self.step_size,
            'nhead': self.nhead,
        } 