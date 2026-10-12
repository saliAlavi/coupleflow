from dataclasses import dataclass
from typing import Tuple
from .baseline_config import BaselineConfig

@dataclass
class HappyCamelConfig(BaselineConfig):
    """Configuration for the HappyCamel model.
    A transformer-based model with position embeddings and optional RoPE attention.
    """
    
    # Model Architecture
    in_channel: int = 64            # Number of input EEG channels
    d_model: int = 768             # Model dimension (was embed_dim)
    d_inner: int = 3072           # Inner dimension of feed-forward network
    n_head: int = 12              # Number of attention heads
    n_layers: int = 12            # Number of transformer layers
    g_con: bool = False           # Whether to use global conditioning
    within_sub_num: int = 71      # Number of subjects for conditioning
    seq_len: int = 640           # Sequence length (10s * 64Hz)
    
    # HappyCamel specific parameters
    use_checkpoint: bool = True    # Whether to use gradient checkpointing
    pe_method: str = 'none'       # Position embedding method
    rope_mode: str = 'shared'     # Rotary position embedding mode
    qkv_bias: bool = False        # Whether to use bias in QKV projections
    qk_scale: float = None        # Scale factor for QK attention
    qk_norm: str = 'layernorm'    # Normalization for QK attention
    norm_layer: str = 'layernorm' # Type of normalization to use

    #optimizer
    warmup_epochs: int = 10
    
    # Override experiment name
    experiment_name: str = "happycamel"
    
    def model_params(self):
        """Return parameters needed for model initialization."""
        return {
            'in_channel': self.in_channel,
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'n_head': self.n_head,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'g_con': self.g_con,
            'within_sub_num': self.within_sub_num,
            'use_checkpoint': self.use_checkpoint,
            'pe_method': self.pe_method,
            'rope_mode': self.rope_mode,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'qk_norm': self.qk_norm,
            'norm_layer': self.norm_layer,
            'seq_len': self.seq_len
        } 