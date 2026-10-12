import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class EnvelopeLSTM(nn.Module):
    """
    LSTM-based model for envelope inpainting.
    Uses stacked LSTM layers with optional dropout and layer normalization.
    """
    
    def __init__(
        self,
        in_channel: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Initial projection for envelope using Conv1d like other models
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3)
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Layer normalization after LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_norm = nn.LayerNorm(lstm_output_dim)
        
        # Final linear projection
        self.output_proj = nn.Linear(lstm_output_dim, 1)
        
        # Save parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
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
            src_key_padding_mask: Optional mask for padding tokens (batch, seq_len) - not used in LSTM
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
        """
        batch_size, seq_len, _ = envelope_context.shape
        device = envelope_context.device

        # Project input to hidden dimension
        x = self.input_proj(envelope_context.transpose(1, 2))  # (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Initialize hidden states
        h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_dim, device=device)
        
        # LSTM forward pass
        x, _ = self.lstm(x, (h0, c0))
        
        # Apply output normalization
        x = self.output_norm(x)
        
        # Final projection
        predictions = self.output_proj(x)

        return {
            'envelope': predictions,
        }


# Example usage:
if __name__ == "__main__":
    # Create random data with shape (batch_size=8, seq_length=320, feature_dim=1)
    batch_size, seq_length, feature_dim = 8, 320, 1
    x = torch.randn(batch_size, seq_length, feature_dim)
    
    # Instantiate the model
    model = EnvelopeLSTM(in_channel=feature_dim, hidden_dim=64, n_layers=1)
    
    # Forward pass
    output = model(eeg=None, envelope_context=x)
    print("Input shape: ", x.shape)      # (8, 320, 1)
    print("Output shape:", output['envelope'].shape) # (8, 320, 1)