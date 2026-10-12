import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class Extractor(nn.Module):
    """Extractor module for the VLAAI model."""
    
    def __init__(
        self,
        filters: Tuple[int, ...] = (256, 256, 256, 128, 128),
        kernels: Tuple[int, ...] = (8, 8, 8, 8, 8),
        input_channels: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if len(filters) != len(kernels):
            raise ValueError("'filters' and 'kernels' must have the same length")
            
        self.layers = nn.ModuleList()
        in_channels = input_channels
        
        # Create convolutional layers
        for filter_, kernel in zip(filters, kernels):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filter_, kernel),
                    # Layer norm over the channel dimension only
                    nn.GroupNorm(1, filter_),  # Equivalent to LayerNorm for the channel dimension
                    nn.LeakyReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filter_
            
        self.kernels = kernels
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the extractor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time)
            
        Returns:
            torch.Tensor: Output tensor
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add zero padding to match TF implementation
            x = F.pad(x, (0, self.kernels[i] - 1))
        return x


class OutputContext(nn.Module):
    """Output context module for the VLAAI model."""
    
    def __init__(
        self,
        filter_: int = 64,
        kernel: int = 32,
        input_channels: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, filter_, kernel),
            nn.GroupNorm(1, filter_),  # Equivalent to LayerNorm for the channel dimension
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.kernel = kernel
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the output context module.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Add zero padding to match TF implementation
        x = F.pad(x, (self.kernel - 1, 0))
        return self.conv(x)


class VLAAI(nn.Module):
    """VLAAI model for EEG envelope decoding."""
    
    def __init__(
        self,
        input_channels: int = 64,
        output_dim: int = 1,
        nb_blocks: int = 4,
        use_skip: bool = True,
        extractor_filters: Tuple[int, ...] = (256, 256, 256, 128, 128),
        extractor_kernels: Tuple[int, ...] = (8, 8, 8, 8, 8),
        context_filter: int = 64,
        context_kernel: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.use_skip = use_skip
        self.nb_blocks = nb_blocks
        
        # Create the extractor and output context models
        self.extractors = nn.ModuleList([
            Extractor(
                filters=extractor_filters,
                kernels=extractor_kernels,
                input_channels=input_channels,
                dropout=dropout
            ) for _ in range(nb_blocks)
        ])
        
        self.dense_layers = nn.ModuleList([
            nn.Linear(extractor_filters[-1], input_channels)
            for _ in range(nb_blocks)
        ])
        
        self.output_contexts = nn.ModuleList([
            OutputContext(
                filter_=context_filter,
                kernel=context_kernel,
                input_channels=input_channels,
                dropout=dropout
            ) for _ in range(nb_blocks)
        ])
        
        # Final output layer
        self.final_dense = nn.Linear(context_filter, output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the VLAAI model.
        
        Args:
            x: Input tensor of shape (batch_size, time, channels)
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch_size, time, output_dim)
        """
        # Transpose for conv1d: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # If using skip connections, start with x set to zero
        if self.use_skip:
            skip = torch.zeros_like(x)
        else:
            skip = x
            
        # Iterate over the blocks
        for i in range(self.nb_blocks):
            if self.use_skip:
                h = self.extractors[i](x + skip)
            else:
                h = self.extractors[i](skip)
                
            # Transpose for dense layer
            h = h.transpose(1, 2)
            h = self.dense_layers[i](h)
            h = h.transpose(1, 2)
            
            skip = self.output_contexts[i](h)
            
        # Final output
        out = self.final_dense(skip.transpose(1, 2))
        
        return {'envelope': out} 