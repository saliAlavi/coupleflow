import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds positional information to the input embeddings.
    Implementation based on the "Attention Is All You Need" paper.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EnvelopeTransformer(nn.Module):
    """
    Transformer-based model for envelope inpainting using PyTorch's TransformerEncoder.
    Includes positional embeddings for better temporal context understanding.
    """
    
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        **kwargs
    ):
        super().__init__()
        
        # Initial projection for envelope
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Using Pre-LN like the original model
        )
        
        # Stack of transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        
        # Final linear projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(
        self,
        eeg: Optional[torch.Tensor] = None,
        envelope_context: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len, channels) - not used in this model
            envelope_context: Input envelope of shape (batch, seq_len, 1)
            src_key_padding_mask: Optional mask for padding tokens (batch, seq_len)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
        """
        batch_size, seq_len, _ = envelope_context.shape
        device = envelope_context.device

        # Project input to model dimension
        x = self.input_proj(envelope_context.transpose(1, 2))  # (batch, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Final projection
        predictions = self.output_proj(x)

        return {
            'envelope': predictions,
        }