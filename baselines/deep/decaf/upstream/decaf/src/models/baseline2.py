import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class BaselineDecoder2(nn.Module):
    """Linear baseline model for envelope decoding using 1D convolution.
    Combines channel outputs using simple averaging.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        filter_length_ms: float = 500,
        sampling_rate: int = 64,
        dropout: float = 0.0
    ):
        """
        Args:
            n_channels (int): Number of EEG channels (default: 64)
            filter_length_ms (float): Filter length in milliseconds (default: 500ms)
            sampling_rate (int): Sampling rate in Hz (default: 64Hz)
            dropout (float): Dropout probability (default: 0.0)
        """
        super().__init__()
        
        # Calculate filter length in samples (500ms = 32 samples at 64Hz)
        self.filter_length = int(filter_length_ms * sampling_rate / 1000)
        
        # Single convolutional layer
        # Input: (batch, channels, time)
        # Output: (batch, channels, time)
        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels,  # One feature per channel
            kernel_size=self.filter_length,
            padding='same',  # Maintain temporal dimension
            groups=n_channels  # Each channel processed independently
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input EEG data of shape (batch_size, sequence_length, channels)
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch_size, sequence_length, 1)
        """
        # Transpose to (batch, channels, time) for convolution
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Average across channels
        x = torch.mean(x, dim=1, keepdim=True)
        
        # Transpose back to (batch, time, channels)
        x = x.transpose(1, 2)
        
        return {'envelope': x}