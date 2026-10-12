import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention
from .utils.modules import FeedForward
from .utils.modules import PE_wrapper
from typing import Optional, Dict


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='layernorm',
                 norm_layer='layernorm',
                 rope_mode='none',
                 use_checkpoint=False):

        super().__init__()
        if norm_layer == 'rmsnorm':
            norm_layer = nn.RMSNorm
            mlp_bias = False
        elif norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
            mlp_bias = True
        else:
            NotImplementedError

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              qk_norm=qk_norm,
                              rope_mode=rope_mode)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim,
                               hidden_size=mlp_hidden_dim,
                               multiple_of=256, bias=mlp_bias)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, x_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              x_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x, x_mask, extras)

    def _forward(self, x, x_mask=None, extras=0):
        # self attention
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, context=None,
                           context_mask=x_mask,
                           extras=extras)
        x = x + x_attn

        # mlp
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        return x


class HappyCamelInpaint(nn.Module):
    """
    Transformer-based model for EEG-envelope inpainting using HappyCamel architecture.
    Concatenates EEG and envelope along channel dimension and learns to reconstruct
    masked portions of the envelope. Uses advanced attention mechanisms and positional
    embeddings from HappyCamel.
    """
    
    def __init__(
        self,
        in_channel: int,
        d_model: int = 768,
        d_inner: int = 3072,
        n_head: int = 12,
        n_layers: int = 12,
        dropout: float = 0.1,
        g_con: bool = False,
        within_sub_num: int = 71,
        use_checkpoint: bool = True,
        pe_method: str = 'rotary',  # Changed default to rotary for better sequence modeling
        rope_mode: str = 'shared',
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        qk_norm: str = 'layernorm',
        norm_layer: str = 'layernorm',
        seq_len: int = 640,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = d_model

        # Initial projection for combined EEG and envelope
        # +1 for the envelope channel like in HappyQuokaInpaint
        self.in_proj = nn.Linear(in_channel + 1, d_model)
        
        # Position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=d_model, method=pe_method,
                             length=seq_len)

        print(f'x position embedding: {pe_method}')
        print(f'rope mode: {self.rope}')

        self.extras = 0

        # Stack of transformer blocks
        block_list = []
        for i in range(n_layers):
            block_list.append(Block(
                dim=d_model,
                num_heads=n_head,
                mlp_ratio=4.,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope_mode=self.rope,
                use_checkpoint=use_checkpoint))
        self.blocks = nn.ModuleList(block_list)

        # Final normalization and projection
        self.norm = nn.LayerNorm(d_model)
        self.final_layer = nn.Linear(d_model, 1)

        # Subject conditioning
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        if g_con:
            self.sub_proj = nn.Linear(self.within_sub_num, d_model)

    def forward(
        self,
        eeg: torch.Tensor,
        envelope_context: torch.Tensor,
        sub_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len, channels)
            envelope_context: Input envelope of shape (batch, seq_len, 1)
            sub_id: Optional subject IDs for global conditioning
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
        """
        batch_size, seq_len, _ = eeg.shape
        device = eeg.device
        
        # Concatenate EEG and envelope context along channel dimension
        x = torch.cat([eeg, envelope_context], dim=-1)
        
        # Project input to model dimension
        x = self.in_proj(x)
        
        # Add positional embeddings
        x = self.x_pe(x)

        # Apply subject conditioning if enabled
        if self.g_con and sub_id is not None:
            # One-hot encode sub_id -> (batch, within_sub_num)
            sub_emb = F.one_hot(sub_id, self.within_sub_num).float()
            # Project one-hot embedding to (batch, d_model)
            sub_emb = self.sub_proj(sub_emb)
            # Broadcast add to each time step
            x = x + sub_emb.unsqueeze(1)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.norm(x)
        x = self.final_layer(x)

        return {
            'envelope': x
        } 