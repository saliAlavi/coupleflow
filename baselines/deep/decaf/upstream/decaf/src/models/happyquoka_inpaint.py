import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from .happyquoka import PreLNFFTBlock

class HappyQuokaInpaint(nn.Module):
    """
    Transformer-based model for EEG-envelope inpainting using Pre-LN FFT blocks.
    Concatenates EEG and envelope along channel dimension and learns to reconstruct
    masked portions of the envelope.
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
        
        # Initial Conv to transform ((in_channel + 1) -> d_model)
        # +1 for the envelope channel
        self.conv = nn.Conv1d(in_channel + 1, d_model, kernel_size=7, padding=3)
        
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
        
        # Final linear projection: (d_model -> 1)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(
        self,
        eeg: torch.Tensor,
        envelope_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len, channels)
            envelope_context: Input envelope of shape (batch, seq_len, 1)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
        """
        batch_size, seq_len, _ = eeg.shape
        device = eeg.device
        
        # Concatenate EEG and normalized masked envelope along channel dimension
        # (batch, seq_len, channels + 1)
        x = torch.cat([eeg, envelope_context], dim=-1)
        
        # Transform input to model dimension
        dec_output = self.conv(x.transpose(1, 2))  # (batch, d_model, time)
        dec_output = dec_output.transpose(1, 2)    # (batch, time, d_model)
        
        # Pass through the PreLNFFTBlock stack
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)
        
        # Final linear projection -> shape (batch, seq_len, 1)
        predictions = self.fc(dec_output)

        return {
            'envelope': predictions
        }