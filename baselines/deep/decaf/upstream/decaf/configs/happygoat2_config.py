from dataclasses import dataclass
from typing import Tuple, List
from .baseline_config import BaselineConfig

@dataclass
class HappyGoat2Config(BaselineConfig):
    """Configuration for the HappyGoat2 model with multi-scale convolutional front end."""
    
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
    multiscale_kernel_sizes: List[int] = (3, 7, 15, 31)  # Kernel sizes for multi-scale conv front end
    
    # Training parameters
    dropout: float = 0.6          # Modified dropout rate
    warmup_epochs: int = 3
    weight_decay: float = 0.0
    
    # Override experiment name
    experiment_name: str = "happygoat2"
    
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
            'within_sub_num': self.within_sub_num,
            'multiscale_kernel_sizes': self.multiscale_kernel_sizes
        } 