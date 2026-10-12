import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Optional, Dict
from .happyquoka import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, PreLNFFTBlock
from .happygoat import TemporalAlignment

class MultiScaleConv(nn.Module):
    """
    Multi-scale convolutional front end that applies parallel convolutions
    with different kernel sizes, then merges them into a single d_model dimension.
    
    Arguments:
    ----------
    in_channel : int
        Number of input channels (e.g., EEG channels).
    d_model : int
        The dimensionality of the transformer model.
    kernel_sizes : list of int
        List of kernel sizes to apply in parallel.
    dropout : float
        Dropout probability applied after merging.
    """
    def __init__(self, in_channel, d_model, kernel_sizes=[3, 7, 15, 32], dropout=0.1):
        super(MultiScaleConv, self).__init__()
        self.kernel_sizes = kernel_sizes
        
        # Each branch will produce some fraction of the final d_model.
        # For simplicity, let's produce d_model // #branches per branch,
        # then we combine them by concatenation. If d_model isn't divisible
        # by #branches, you can handle that with a final linear or rounding.
        self.num_branches = len(kernel_sizes)
        branch_dim = d_model // self.num_branches
        
        # Define parallel convolutions
        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channel, branch_dim,
                kernel_size=k,
                padding=k // 2  # keep output length the same
            )
            for k in kernel_sizes
        ])
        
        # If d_model is not exactly num_branches * branch_dim, 
        # we’ll do an additional linear projection to d_model.
        # But if it matches, you can skip this.
        self.final_proj = None
        if branch_dim * self.num_branches != d_model:
            self.final_proj = nn.Linear(branch_dim * self.num_branches, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x : (batch, time, in_channel)
        
        Returns:
            out : (batch, time, d_model)
        """
        # Switch to (batch, in_channel, time) for conv
        x = x.transpose(1, 2)  # (B, in_channel, T)
        
        # Parallel convolutions
        branch_outputs = []
        for conv in self.branches:
            branch_out = conv(x)  # (B, branch_dim, T)
            branch_outputs.append(branch_out)
        
        # Concatenate along channel dimension
        merged = torch.cat(branch_outputs, dim=1)  # (B, sum_of_branch_dims, T)
        
        # Optional final projection
        if self.final_proj is not None:
            merged = merged.transpose(1, 2)  # (B, T, C)
            merged = self.final_proj(merged) # (B, T, d_model)
            merged = merged.transpose(1, 2)  # (B, d_model, T)
        else:
            # For consistency, rename merged for clarity
            merged = merged

        merged = self.dropout(merged)
        
        # Switch back to (batch, time, d_model)
        out = merged.transpose(1, 2)
        return out



###############################################################################
# 5) Decoder
###############################################################################
class HappyGoat2(nn.Module):
    """
    Transformer-based model for EEG envelope decoding using a Multi-Scale 
    Convolutional front end + Pre-LN FFT blocks. 
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
                 multiscale_kernel_sizes=[3,7,15,31],
                 **kwargs):
        super(HappyGoat2, self).__init__()

        self.g_con = g_con
        self.within_sub_num = within_sub_num

        # Multi-scale front end replaces the single conv layer
        self.frontend = MultiScaleConv(
            in_channel=in_channel,
            d_model=d_model,
            kernel_sizes=multiscale_kernel_sizes,
            dropout=dropout
        )

        # Stack of Pre-LN FFT Blocks (same as in original code)
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

    def forward(self, x: torch.Tensor, sub_id: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch_size, time, in_channel)
            sub_id: optional (batch_size,)
        
        Returns:
            Dict with key 'envelope' -> (batch_size, time, 1)
        """
        # 1) Multi-scale front end
        dec_output = self.frontend(x)  # (batch, time, d_model)

        # 2) Optional global conditioning
        if self.g_con and sub_id is not None:
            sub_emb = F.one_hot(sub_id, self.within_sub_num).float()  # (batch, within_sub_num)
            sub_emb = self.sub_proj(sub_emb)                          # (batch, d_model)
            dec_output = dec_output + sub_emb.unsqueeze(1)           # add to each time step

        # 3) Pass through the PreLNFFTBlock stack
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)

        # 4) Final linear projection
        output = self.fc(dec_output)

        return {'envelope': output}
