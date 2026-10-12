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


class HappyCamel(nn.Module):
    """Transformer-based model for EEG envelope decoding using Pre-LN blocks.
    Features multi-head attention, position embeddings, and optional subject-specific conditioning.
    """
    
    def __init__(self,
                 in_channel,  # Changed from in_chans to match HappyQuoka
                 d_model=768,  # Renamed from embed_dim to match convention
                 d_inner=3072,  # Added to match HappyQuoka's d_inner
                 n_head=12,    # Renamed from num_heads
                 n_layers=12,  # Renamed from depth
                 dropout=0.1,
                 g_con=False,
                 within_sub_num=71,
                 use_checkpoint=True,
                 pe_method='none',
                 rope_mode='shared',
                 qkv_bias=False,
                 qk_scale=None,
                 qk_norm='layernorm',
                 norm_layer='layernorm',
                 seq_len=640,  # Added sequence length parameter (10s * 64Hz)
                 **kwargs
                ):
        super().__init__()
        self.num_features = self.embed_dim = d_model

        # input projection
        self.in_proj = nn.Linear(in_channel, d_model)
        
        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=d_model, method=pe_method,
                             length=seq_len)  # Use standard sequence length

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
        self.final_layer = nn.Linear(d_model, 1)  # Changed to output 1 channel like HappyQuoka

        # Subject conditioning
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        if g_con:
            self.sub_proj = nn.Linear(self.within_sub_num, d_model)

    def forward(self, x: torch.Tensor, sub_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, time, channels)
            sub_id: Optional subject IDs of shape (batch_size,) for global conditioning
            
        Returns:
            Dict containing:
                'envelope': Predicted envelope of shape (batch_size, time, 1)
        """
        # Project input to model dimension
        x = self.in_proj(x)
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

        return {'envelope': x}


