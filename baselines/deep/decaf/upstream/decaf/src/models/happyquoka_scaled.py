import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Optional, Dict

###############################################################################
# 1) Scaled Dot-Product Attention
###############################################################################
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, len, d_k or d_v)
        # mask: optional boolean mask of shape (batch, len_q, len_k)

        attn = torch.bmm(q, k.transpose(1, 2))  # (batch, len_q, len_k)
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (batch, len_q, d_v)

        return output, attn


###############################################################################
# 2) Multi-Head Attention
###############################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Linear transformations for q, k, v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5)
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Final linear after concatenating multi-head outputs
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, seq_len, d_model)
        # mask: optional boolean mask of shape (batch, seq_len, seq_len)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # Save residual
        residual = q

        # Pre-norm
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # Linear projections: (batch, seq_len, n_head*d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose to shape (n_head*batch, seq_len, d_k or d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # Apply scaled dot-product attention
        output, attn = self.attention(q, k, v, mask=mask)

        # Reshape back to (batch, seq_len, n_head*d_v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1
        )  # (batch, seq_len, n_head*d_v)

        # Final linear + dropout
        output = self.dropout(self.fc(output))

        # Add residual connection
        output = output + residual

        return output, attn


###############################################################################
# 3) Positionwise Feed-Forward
###############################################################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout=0.1):
        super().__init__()

        # Conv1d for position-wise feed-forward
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0],
            padding=fft_conv1d_padding[0]
        )
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1],
            padding=fft_conv1d_padding[1]
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_in)
        residual = x

        # Pre-norm
        x = self.layer_norm(x)

        # Switch to (batch, d_in, seq_len) for Conv1d
        output = x.transpose(1, 2)

        # Two Conv1D layers with ReLU in between
        output = self.w_1(output)
        output = F.relu(output)
        output = self.w_2(output)

        # Switch back to (batch, seq_len, d_in)
        output = output.transpose(1, 2)

        output = self.dropout(output)
        output = output + residual

        return output


###############################################################################
# 4) Pre-LN FFT Block
###############################################################################
class PreLNFFTBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 **kwargs):
        super(PreLNFFTBlock, self).__init__()

        d_k = d_model // n_head
        d_v = d_model // n_head

        # Multi-Head Attention sub-layer
        self.slf_attn = MultiHeadAttention(
            n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout
        )
        # Position-wise Feed-Forward sub-layer
        self.pos_ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout
        )

    def forward(self, fft_input):
        # fft_input: (batch, seq_len, d_model)

        # Self-attention
        fft_output, _ = self.slf_attn(fft_input, fft_input, fft_input)

        # Position-wise FFN
        fft_output = self.pos_ffn(fft_output)

        return fft_output


###############################################################################
# 5) Decoder
###############################################################################
class HappyQuokaScaled(nn.Module):
    """Transformer-based model for EEG envelope decoding using Pre-LN FFT blocks.
    Features multi-head attention, optional subject-specific conditioning, and
    output scaling to ensure values are always between -1 and 1.
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
        super(HappyQuokaScaled, self).__init__()

        self.g_con = g_con
        self.within_sub_num = within_sub_num

        # Initial Conv to transform (in_channel -> d_model)
        self.conv = nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3)

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
                'envelope': Predicted envelope of shape (batch_size, time, 1) scaled to [-1, 1]
        """
        # Transform input channels to model dimension
        dec_output = self.conv(x.transpose(1, 2))  # (batch, d_model, time)
        dec_output = dec_output.transpose(1, 2)    # (batch, time, d_model)

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
        
        # Apply tanh to constrain output between -1 and 1
        # output = torch.tanh(output)

        # Scale the output to be between 0 and 1 using sigmoid
        output = torch.sigmoid(output)

        return {'envelope': output, 'dec_output': dec_output} 