from dataclasses import dataclass
from typing import Tuple, Literal, ClassVar
from .baseline_config import BaselineConfig

@dataclass
class TwoBranchBaseConfig(BaselineConfig):
    """Base configuration for two-branch models.
    Combines HappyQuoka (EEG branch) and EnvelopeForecaster (envelope branch) with flexible fusion strategies.
    """
    
    # Available strategies and context durations
    FUSION_STRATEGIES: ClassVar[Tuple[str, ...]] = ('average', 'weighted', 'dynamic', 'early_f', 'allow_one_branch', 'envelope_only')
    CONTEXT_SECONDS: ClassVar[Tuple[float, ...]] = (0.0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.5, 1.0, 1.5, 3.0)
    
    # Model Architecture Configuration
    in_channel: int = 64            # Number of input EEG channels
    d_model: int = 128             # Model dimension
    d_inner: int = 1024           # Inner dimension of feed-forward network
    n_head: int = 2               # Number of attention heads
    n_layers: int = 8             # Number of transformer layers
    fft_conv1d_kernel: Tuple[int, int] = (9, 1)  # Kernel sizes for FFT conv layers
    fft_conv1d_padding: Tuple[int, int] = (4, 0)  # Padding for FFT conv layers
    
    # Training Configuration
    dropout: float = 0.6          # Dropout rate
    warmup_epochs: int = 3        # Number of warmup epochs for transformer schedule
    weight_decay: float = 0.0     # Weight decay for optimizer
    learning_rate: float = 3e-4   # Base learning rate for branches trained from scratch
    finetune_lr: float = 1e-7    # Learning rate for fine-tuned branches
    needs_transformer_schedule: bool = True
    
    # Contextual Configuration
    is_contextual: bool = True  # Flag for contextual models (e.g., inpainting models)
    context_mask_strategy: Literal['deterministic', 'random'] = "none"  # Options: "deterministic", "random", "none"
    
    # Fusion Configuration
    fusion_strategy: Literal['average', 'weighted', 'dynamic', 'early_f', 'allow_one_branch', 'envelope_only'] = 'weighted'
    
    # EnvelopeForecaster Configuration
    rnn_type: Literal['lstm', 'gru', 'transformer'] = 'lstm'  # Type of recurrent model to use
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'in_channel': self.in_channel,
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'n_head': self.n_head,
            'n_layers': self.n_layers,
            'fft_conv1d_kernel': self.fft_conv1d_kernel,
            'fft_conv1d_padding': self.fft_conv1d_padding,
            'dropout': self.dropout,
            'fusion_strategy': self.fusion_strategy,
            'rnn_type': self.rnn_type,
            'context_seconds': self.context_seconds,
            'prediction_seconds': self.prediction_seconds
        }