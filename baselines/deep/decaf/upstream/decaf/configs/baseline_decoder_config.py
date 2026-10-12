from dataclasses import dataclass
from .baseline_config import BaselineConfig

@dataclass
class BaselineDecoderConfig(BaselineConfig):
    """Configuration for BaselineDecoder model."""
    
    # Model Configuration
    n_channels: int = 64
    filter_length_ms: float = 500
    sampling_rate: int = 64
    dropout: float = 0.2
    
    # Override experiment name
    experiment_name: str = "baseline_decoder"
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'n_channels': self.n_channels,
            'filter_length_ms': self.filter_length_ms,
            'sampling_rate': self.sampling_rate,
            'dropout': self.dropout
        } 