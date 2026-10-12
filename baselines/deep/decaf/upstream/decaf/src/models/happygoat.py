import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Optional, Dict
from .happyquoka import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, PreLNFFTBlock

class TemporalAlignment(nn.Module):
    def __init__(self, d_model, max_shift=50):
        super().__init__()
        self.max_shift = max_shift
        self.shift_pred = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=15, padding=7)
        )
        
    def forward(self, x):
        """Differentiable temporal alignment using learned shifts"""
        batch_size, seq_len, d_model = x.shape
        
        # Predict shift in samples for each instance (batch_size, 1)
        shift_logits = self.shift_pred(x.permute(0,2,1))  # (batch_size, 1, seq_len)
        shifts = torch.tanh(shift_logits.mean(dim=2)) * self.max_shift  # (batch_size, 1)

        # Create affine transformation matrices (batch_size, 2, 3)
        theta = torch.zeros(batch_size, 2, 3, device=x.device)
        theta[:, 0, 0] = 1  # Identity scale for x-axis
        theta[:, 1, 1] = 1  # Identity scale for y-axis
        theta[:, 0, 2] = (2*shifts.squeeze())/seq_len  # Normalized x-shift

        # Generate sampling grid (batch_size, 1, seq_len, 2)
        grid = F.affine_grid(
            theta, 
            [batch_size, d_model, 1, seq_len],
            align_corners=False
        )

        # Apply grid sampling (batch_size, d_model, 1, seq_len)
        x_aligned = F.grid_sample(
            x.permute(0,2,1).unsqueeze(2),  # (batch, d_model, 1, seq_len)
            grid,
            padding_mode='border',
            align_corners=False
        )
        
        return x_aligned.squeeze(2).permute(0,2,1)  # (batch, seq_len, d_model)


###############################################################################
# 5) Decoder
###############################################################################
class HappyGoat(nn.Module):
    """Transformer-based model for EEG envelope decoding using Pre-LN FFT blocks.
    Features multi-head attention and optional subject-specific conditioning.
    """
    
    def __init__(self,
                 in_channel,
                 d_model,
                 d_inner,
                 n_head,
                 n_layers,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 g_con,
                 within_sub_num=71,
                 **kwargs):
        super(HappyGoat, self).__init__()

        self.g_con = g_con
        self.within_sub_num = within_sub_num

        # Initial Conv to transform (in_channel -> d_model)
        self.conv = nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3)
        self.temp_align = TemporalAlignment(d_model, max_shift=50)

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

        # Optional global conditioning sub-projection
        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

        # Final linear projection: (d_model -> 1)
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
        # Transform input channels to model dimension
        dec_output = self.conv(x.transpose(1, 2))  # (batch, d_model, time)
        dec_output = dec_output.transpose(1, 2)    # (batch, time, d_model)

        dec_output = self.temp_align(dec_output)

        # Optional global conditioning
        if self.g_con and sub_id is not None:
            # One-hot encode sub_id -> (batch, within_sub_num)
            sub_emb = F.one_hot(sub_id, self.within_sub_num).float()
            # Project one-hot embedding to (batch, d_model)
            sub_emb = self.sub_proj(sub_emb)
            # Broadcast add to each time step
            dec_output = dec_output + sub_emb.unsqueeze(1)

        # Pass through the PreLNFFTBlock stack
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)

        # Final linear projection -> shape (batch, seq_len, 1)
        output = self.fc(dec_output)

        return {'envelope': output}
