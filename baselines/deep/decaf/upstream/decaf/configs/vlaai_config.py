from dataclasses import dataclass
from typing import Tuple, List
from .baseline_config import BaselineConfig

@dataclass
class VlaaiConfig(BaselineConfig):
    """Configuration for VLAAI model."""
    
    # Model Configuration
    n_channels: int = 64
    output_dim: int = 1
    nb_blocks: int = 4
    use_skip: bool = True
    
    # Extractor Configuration
    extractor_filters: Tuple[int, ...] = (256, 256, 256, 128, 128)
    extractor_kernels: Tuple[int, ...] = (8, 8, 8, 8, 8)
    
    # Output Context Configuration
    context_filter: int = 64
    context_kernel: int = 32
    
    # Override experiment name
    experiment_name: str = "vlaai"
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'input_channels': self.n_channels,
            'output_dim': self.output_dim,
            'nb_blocks': self.nb_blocks,
            'use_skip': self.use_skip,
            'extractor_filters': self.extractor_filters,
            'extractor_kernels': self.extractor_kernels,
            'context_filter': self.context_filter,
            'context_kernel': self.context_kernel,
            'dropout': self.dropout
        } 