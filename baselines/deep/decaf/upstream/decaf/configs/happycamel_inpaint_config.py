from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal, Optional
from .baseline_config import BaselineConfig

@dataclass
class HappyCamelInpaintConfig(BaselineConfig):
    """Configuration for HappyCamelInpaint model.
    Combines HappyCamel4 architecture with inpainting capabilities.
    """
    
    # Available context durations and strategies
    CONTEXT_SECONDS: ClassVar[Tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    MASK_STRATEGIES: ClassVar[Tuple[str, ...]] = ('deterministic', 'random')
    
    # Model Architecture (same as HappyCamel4)
    in_channel: int = 64            # Number of input EEG channels
    d_model: int = 768             # Model dimension
    d_inner: int = 3072           # Inner dimension of feed-forward network
    n_head: int = 12              # Number of attention heads
    n_layers: int = 12            # Number of transformer layers
    g_con: bool = False           # Whether to use global conditioning
    within_sub_num: int = 71      # Number of subjects for conditioning
    seq_len: int = 640           # Sequence length (10s * 64Hz)
    
    # HappyCamel specific parameters (same as HappyCamel4)
    use_checkpoint: bool = True    # Whether to use gradient checkpointing
    pe_method: str = 'none'       # Position embedding method
    rope_mode: str = 'shared'     # Rotary position embedding mode
    qkv_bias: bool = False        # Whether to use bias in QKV projections
    qk_scale: Optional[float] = None  # Scale factor for QK attention
    qk_norm: str = 'layernorm'    # Normalization for QK attention
    norm_layer: str = 'layernorm' # Type of normalization to use
    
    # Training Configuration (same as HappyCamel4)
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    dropout: float = 0.1
    warmup_epochs: int = 20
    needs_transformer_schedule: bool = True
    
    # Contextual Configuration (specific to inpainting)
    is_contextual: bool = True
    context_mask_strategy: Literal['deterministic', 'random'] = "deterministic"
    context_seconds: float = 1.0
    mask_value: float = 0.0
    
    # Experiment Configuration
    experiment_name: str = "happycamel_inpaint"
    autoregressive: bool = False
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Validate configuration and set experiment name."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
    
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