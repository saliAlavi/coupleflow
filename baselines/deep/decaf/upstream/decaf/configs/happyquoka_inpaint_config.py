from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal
from .baseline_config import BaselineConfig

@dataclass
class HappyQuokaInpaintConfig(BaselineConfig):
    """Configuration for HappyQuokaInpaint model."""

    # Available context durations and strategies
    CONTEXT_SECONDS: ClassVar[Tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    MASK_STRATEGIES: ClassVar[Tuple[str, ...]] = ('deterministic', 'random')
    
    # Model Architecture Configuration
    in_channel: int = 64            # Number of input EEG channels
    d_model: int = 128             # Model dimension (same as HappyQuoka4)
    d_inner: int = 1024           # Inner dimension of feed-forward network
    n_head: int = 2               # Number of attention heads (same as HappyQuoka4)
    n_layers: int = 8             # Number of transformer layers (same as HappyQuoka4)
    fft_conv1d_kernel: Tuple[int, int] = (9, 1)  # Kernel sizes for FFT conv layers
    fft_conv1d_padding: Tuple[int, int] = (4, 0)  # Padding for FFT conv layers
    
    # Training Configuration
    dropout: float = 0.6          # Modified dropout rate (same as HappyQuoka4)
    warmup_epochs: int = 3        # Same as HappyQuoka4
    weight_decay: float = 0.0     # Same as HappyQuoka4
    needs_transformer_schedule: bool = True
    
    # Contextual Configuration
    is_contextual: bool = True  # Flag for contextual models (e.g., inpainting models)
    context_mask_strategy: Literal['deterministic', 'random'] = "deterministic"  # Masking strategy
    context_seconds: float = 1.0  # Number of seconds to mask for context (ignored when strategy is random)
    mask_value: float = 0.0  # Value to mask the context with
        
    # Experiment Configuration
    experiment_name: str = "happyquoka_inpaint"
    
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
        } 