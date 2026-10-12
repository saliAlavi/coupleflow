import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class BaselineDecoder(nn.Module):
    """Linear baseline model for envelope decoding using 1D convolution.
    Combines channel outputs using learnable projection.
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
            padding='same',  # Explicit padding
            groups=n_channels  # Each channel processed independently
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection to combine channel outputs
        self.projection = nn.Conv1d(
            in_channels=n_channels,
            out_channels=1,  # Final single channel output
            kernel_size=1,  # 1x1 convolution
            bias=True
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, channels)
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch, time, 1)
        """
        # Transpose for conv1d: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Project to single channel
        x = self.projection(x)
        
        # Transpose back: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        return {'envelope': x}

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example usage
    batch_size = 32
    sequence_length = 640  # 10 seconds at 64 Hz
    n_channels = 64
    
    # Create test data
    x = torch.randn(batch_size, sequence_length, n_channels)
    
    print("\nTesting Baseline Model (with learnable projection):")
    # Create model 1
    model1 = BaselineDecoder(
        n_channels=n_channels,
        filter_length_ms=500,
        sampling_rate=64,
        dropout=0.2
    )
    
    # Forward pass
    with torch.no_grad():
        output1 = model1(x)
    
    print(f"Model 1 Parameters: {count_parameters(model1)}")
    print(f"Conv parameters: {sum(p.numel() for p in model1.conv.parameters())}")
    print(f"Projection parameters: {sum(p.numel() for p in model1.projection.parameters())}")
    print(f"Output shape: {output1['envelope'].shape}")
    
    print("\nTesting Baseline Model 2 (with averaging):")
    # Create model 2
    model2 = BaselineDecoder2(
        n_channels=n_channels,
        filter_length_ms=500,
        sampling_rate=64,
        dropout=0.2
    )
    
    # Forward pass
    with torch.no_grad():
        output2 = model2(x)
    
    print(f"Model 2 Parameters: {count_parameters(model2)}")
    print(f"Conv parameters: {sum(p.numel() for p in model2.conv.parameters())}")
    print(f"Output shape: {output2['envelope'].shape}")
    
    # Verify shapes
    assert output1['envelope'].shape == output2['envelope'].shape == (batch_size, sequence_length, 1), \
        "Output shapes don't match expected shape"
    
    print("\nShape verification passed!") 