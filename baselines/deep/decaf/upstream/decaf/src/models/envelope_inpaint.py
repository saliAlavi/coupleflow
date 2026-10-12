import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from .happyquoka import PreLNFFTBlock

class EnvelopeInpaint(nn.Module):
    """
    Transformer-based model for envelope inpainting using Pre-LN FFT blocks.
    Focuses solely on predicting masked portions of the envelope from context.
    """
    
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        n_layers: int,
        fft_conv1d_kernel: list,
        fft_conv1d_padding: list,
        dropout: float,
        **kwargs
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        
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
        
        # Final linear projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(
        self,
        eeg: torch.Tensor = None,
        envelope_context: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len, channels) - not used in this model
            envelope_context: Input envelope of shape (batch, seq_len, 1)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
        """
        batch_size, seq_len, _ = envelope_context.shape
        device = envelope_context.device

        # Project input to model dimension
        x = self.input_proj(envelope_context.transpose(1, 2))  # (batch, 1, seq_len) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        
        for layer in self.layer_stack:
            x = layer(x)

        # Final projection
        predictions = self.output_proj(x)

        return {
            'envelope': predictions,
        }