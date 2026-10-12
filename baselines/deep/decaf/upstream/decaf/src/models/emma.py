import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from .happyquoka import PreLNFFTBlock, MultiHeadAttention, PositionwiseFeedForward

class MultiRateStream(nn.Module):
    """Single stream of the multi-rate model with specific kernel size."""
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        n_layers: int,
        kernel_size: int,
        fft_conv1d_kernel: List[int],
        fft_conv1d_padding: List[int],
        dropout: float
    ):
        super().__init__()
        
        # Initial Conv with specified kernel size
        self.conv = nn.Conv1d(
            in_channel, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        # Stack of Pre-LN FFT Blocks
        self.layer_stack = nn.ModuleList([
            PreLNFFTBlock(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                fft_conv1d_kernel=fft_conv1d_kernel,
                fft_conv1d_padding=fft_conv1d_padding,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, time, channels)
            
        Returns:
            Tensor of shape (batch_size, time, d_model)
        """
        # Transform input channels to model dimension
        out = self.conv(x.transpose(1, 2))  # (batch, d_model, time)
        out = out.transpose(1, 2)    # (batch, time, d_model)
        
        # Pass through transformer blocks
        for layer in self.layer_stack:
            out = layer(out)
            
        return out

class EMMA(nn.Module):
    """Multi-rate transformer model for EEG envelope decoding.
    Features three parallel streams with different kernel sizes and optional
    subject-specific conditioning.
    """
    
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        n_layers: int,
        fft_conv1d_kernel: List[int],
        fft_conv1d_padding: List[int],
        dropout: float,
        g_con: bool,
        within_sub_num: int = 71,
        kernel_sizes: List[int] = [7, 15, 31],
        **kwargs
    ):
        super().__init__()
        
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.d_model = d_model
        
        # Create three parallel streams with different kernel sizes
        self.streams = nn.ModuleList([
            MultiRateStream(
                in_channel=in_channel,
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                n_layers=n_layers,
                kernel_size=k,
                fft_conv1d_kernel=fft_conv1d_kernel,
                fft_conv1d_padding=fft_conv1d_padding,
                dropout=dropout
            ) for k in kernel_sizes
        ])
        
        # Optional global conditioning sub-projection
        if g_con:
            self.sub_proj = nn.Linear(within_sub_num, d_model)
        
        # Final layers after concatenation
        self.fusion_layer = nn.Sequential(
            nn.Linear(len(kernel_sizes) * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, sub_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, time, channels)
            sub_id: Optional subject IDs of shape (batch_size,) for global conditioning
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch_size, time, 1)
        """
        # Process each stream
        stream_outputs = []
        for stream in self.streams:
            stream_out = stream(x)
            
            # Apply global conditioning if enabled
            if self.g_con and sub_id is not None:
                # One-hot encode sub_id -> (batch, within_sub_num)
                sub_emb = F.one_hot(sub_id, self.within_sub_num).float()
                # Project one-hot embedding to (batch, d_model)
                sub_emb = self.sub_proj(sub_emb)
                # Broadcast add to each time step
                stream_out = stream_out + sub_emb.unsqueeze(1)
                
            stream_outputs.append(stream_out)
        
        # Concatenate stream outputs along feature dimension
        concat_output = torch.cat(stream_outputs, dim=-1)  # (batch, time, 3*d_model)
        
        # Fuse features
        fused = self.fusion_layer(concat_output)  # (batch, time, d_model)
        
        # Final projection
        output = self.fc(fused)  # (batch, time, 1)
        
        return {'envelope': output}
