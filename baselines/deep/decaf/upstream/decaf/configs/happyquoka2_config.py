from dataclasses import dataclass
from typing import Tuple
from .baseline_config import BaselineConfig

@dataclass
class HappyQuoka2Config(BaselineConfig):
    """Configuration for the HappyQuoka2 model with modified hyperparameters."""
    
    # Model Architecture
    in_channel: int = 64            # Number of input EEG channels
    d_model: int = 128             # Model dimension (reduced from 256)
    d_inner: int = 1024           # Inner dimension of feed-forward network
    n_head: int = 2               # Number of attention heads (reduced from 8)
    n_layers: int = 8             # Number of transformer layers (increased from 4)
    fft_conv1d_kernel: Tuple[int, int] = (9, 1)  # Kernel sizes for FFT conv layers
    fft_conv1d_padding: Tuple[int, int] = (4, 0)  # Padding for FFT conv layers
    g_con: bool = False             # Whether to use global conditioning
    within_sub_num: int = 71       # Number of subjects for conditioning
    
    # Training parameters
    dropout: float = 0.3          # Modified dropout rate
    
    # Override experiment name
    experiment_name: str = "happyquoka2"
    
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
            'g_con': self.g_con,
            'within_sub_num': self.within_sub_num
        } 