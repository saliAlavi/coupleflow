import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

from .vlaai import Extractor
from .happyquoka import MultiHeadAttention, PositionwiseFeedForward, PreLNFFTBlock

class ExtractorFeedForward(nn.Module):
    """Feed-forward module using VLAAI's extractor instead of Conv1d layers."""
    
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        
        # First extractor for expanding dimension
        self.extractor1 = Extractor(
            filters=(d_hid,),  # Expand to d_hid
            kernels=(9,),      # Using similar kernel size as original
            input_channels=d_in,
            dropout=dropout
        )
        
        # Second extractor for reducing dimension
        self.extractor2 = Extractor(
            filters=(d_in,),   # Reduce back to d_in
            kernels=(1,),      # Point-wise conv equivalent
            input_channels=d_hid,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_in)
        residual = x
        
        # Pre-norm
        x = self.layer_norm(x)
        
        # Switch to (batch, d_in, seq_len) for extractors
        x = x.transpose(1, 2)
        
        # First extractor expands dimension
        x = self.extractor1(x)
        x = F.relu(x)
        
        # Second extractor reduces dimension
        x = self.extractor2(x)
        
        # Switch back to (batch, seq_len, d_in)
        x = x.transpose(1, 2)
        
        x = self.dropout(x)
        x = x + residual
        
        return x

class ExtractorFFTBlock(nn.Module):
    """Modified PreLNFFTBlock that uses ExtractorFeedForward instead of PositionwiseFeedForward."""
    
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        dropout: float = 0.1,
        **kwargs  # To maintain compatibility with original PreLNFFTBlock
    ):
        super().__init__()
        
        d_k = d_model // n_head
        d_v = d_model // n_head
        
        # Multi-Head Attention (same as original)
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        
        # ExtractorFeedForward instead of PositionwiseFeedForward
        self.pos_ffn = ExtractorFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x, _ = self.slf_attn(x, x, x)
        
        # Extractor feed-forward
        x = self.pos_ffn(x)
        
        return x

class VLAAIQuoka(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,            # Number of input EEG channels
        d_model: int = 128,              # Model dimension (reduced from 256)
        d_inner: int = 1024,            # Inner dimension of feed-forward network
        n_head: int = 2,                # Number of attention heads (reduced from 8)
        n_layers: int = 8,              # Number of transformer layers
        dropout: float = 0.3,           # Modified dropout rate
        extractor_filters: Tuple[int, ...] = (256, 256, 256, 128, 128),
        extractor_kernels: Tuple[int, ...] = (8, 8, 8, 8, 8),
        g_con: bool = False,            # Whether to use global conditioning
        within_sub_num: int = 71,       # Number of subjects for conditioning
        use_extractor_fft: bool = True,  # Whether to use ExtractorFFTBlock instead of PreLNFFTBlock
        fft_conv1d_kernel: Tuple[int, int] = (9, 1),  # Kernel sizes for FFT conv layers (only used if use_extractor_fft=False)
        fft_conv1d_padding: Tuple[int, int] = (4, 0),  # Padding for FFT conv layers (only used if use_extractor_fft=False)
    ):
        super().__init__()
        
        # VLAAI's extractor instead of simple conv
        self.extractor = Extractor(
            filters=extractor_filters,
            kernels=extractor_kernels,
            input_channels=in_channel,
            dropout=dropout
        )
        
        # Project to d_model dimension
        self.input_projection = nn.Linear(extractor_filters[-1], d_model)
        
        # Stack of FFT Blocks (either PreLN or Extractor-based)
        block_class = ExtractorFFTBlock if use_extractor_fft else PreLNFFTBlock
        self.layer_stack = nn.ModuleList([
            block_class(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                dropout=dropout,
                **({"fft_conv1d_kernel": fft_conv1d_kernel, 
                    "fft_conv1d_padding": fft_conv1d_padding} if not use_extractor_fft else {})
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Global conditioning
        self.g_con = g_con
        if g_con:
            self.subject_embedding = nn.Embedding(within_sub_num, d_model)
        
    def forward(self, x: torch.Tensor, sub_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, time, channels)
            sub_id: Optional subject ID tensor for global conditioning
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch_size, time, 1)
        """
        # Extract features using VLAAI's extractor
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.extractor(x)  # (batch, filters[-1], time)
        x = x.transpose(1, 2)  # (batch, time, filters[-1])
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, time, d_model)
        
        # Add subject embedding if using global conditioning
        if self.g_con and sub_id is not None:
            sub_emb = self.subject_embedding(sub_id)  # (batch, d_model)
            x = x + sub_emb.unsqueeze(1)  # Add to all time steps
        
        # Pass through FFT blocks
        for layer in self.layer_stack:
            x = layer(x)
            
        # Output projection
        output = self.output_projection(x)  # (batch, time, 1)
        
        return {'envelope': output} 