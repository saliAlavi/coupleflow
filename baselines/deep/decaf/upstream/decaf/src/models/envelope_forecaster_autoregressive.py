import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math
import numpy as np

class EnvelopeForecasterAutoregressive(nn.Module):
    """
    Autoregressive model for envelope forecasting.
    Takes 2 seconds of context and predicts multiple future steps autoregressively.
    
    With a sampling rate of 64 Hz:
    - 2 seconds context = 128 samples
    - Predicts multiple steps into the future during training
    """
    
    def __init__(
        self,
        in_channel: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "lstm",  # Options: "lstm", "gru", "transformer"
        bidirectional: bool = False,
        context_seconds: float = 2.0,
        prediction_seconds: float = 1.0,
        future_seconds: float = 8.0,  # Total future seconds to predict during training
        nhead: int = 4,  # Number of attention heads for transformer
        step_size: float = 0.5,  # Step size in seconds for autoregressive prediction
        **kwargs
    ):
        super().__init__()
        
        # Save parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.sr = 64  # Sampling rate
        
        # Calculate context and prediction lengths
        self.context_len = int(context_seconds * self.sr)  # 2 seconds at 64 Hz = 128 samples
        self.pred_len = int(prediction_seconds * self.sr)  # 1 second at 64 Hz = 64 samples
        self.future_len = int(future_seconds * self.sr)    # 8 seconds at 64 Hz = 512 samples
        self.step_size = int(step_size * self.sr)          # Step size for autoregressive prediction
        
        # Initial projection for envelope using Conv1d
        self.input_proj = nn.Conv1d(in_channel, hidden_dim, kernel_size=7, padding=3)
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Define the model architecture based on rnn_type
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            # Output dimension after RNN
            rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            # Output dimension after RNN
            rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
        elif self.rnn_type == "transformer":
            # Positional encoding for transformer
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
            
            # Transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            
            # Transformer encoder
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layers
            )
            
            # Output dimension after transformer
            rnn_output_dim = hidden_dim
            
        else:
            raise ValueError(f"Unsupported model type: {rnn_type}. Choose 'lstm', 'gru', or 'transformer'.")
        
        # Layer normalization after model
        self.output_norm = nn.LayerNorm(rnn_output_dim)
        
        # Attention mechanism for better context utilization (for LSTM and GRU)
        if self.rnn_type in ["lstm", "gru"]:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_output_dim,
                num_heads=nhead,
                batch_first=True
            )
        
        # Final projection layers
        self.fc1 = nn.Linear(rnn_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.pred_len)
        
    def _single_step_forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Perform a single step forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
            hidden_states: Hidden states for RNN models
            src_key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Tuple containing:
                - Prediction tensor of shape (batch, pred_len, 1)
                - Updated hidden states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project input to hidden dimension
        x_proj = self.input_proj(x.transpose(1, 2))  # (batch, hidden_dim, seq_len)
        x_proj = x_proj.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Apply input normalization
        x_proj = self.input_norm(x_proj)
        
        # Process based on model type
        if self.rnn_type == "lstm":
            # Initialize hidden states if not provided
            if hidden_states is None:
                h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                                batch_size, self.hidden_dim, device=device)
                c0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                                batch_size, self.hidden_dim, device=device)
                hidden_states = (h0, c0)
            
            # LSTM forward pass
            x_out, hidden_states = self.rnn(x_proj, hidden_states)
            
            # Apply output normalization
            x_out = self.output_norm(x_out)
            
            # Apply self-attention to better capture context dependencies
            attn_output, _ = self.attention(x_out, x_out, x_out)
            
            # Get the last hidden state for prediction
            if self.bidirectional:
                # For bidirectional, concatenate the last hidden states from both directions
                last_hidden = torch.cat([hidden_states[0][-2], hidden_states[0][-1]], dim=1)
            else:
                # For unidirectional, use the last hidden state
                last_hidden = hidden_states[0][-1]
                
        elif self.rnn_type == "gru":
            # Initialize hidden states if not provided
            if hidden_states is None:
                h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                                batch_size, self.hidden_dim, device=device)
                hidden_states = h0
            
            # GRU forward pass
            x_out, hidden_states = self.rnn(x_proj, hidden_states)
            
            # Apply output normalization
            x_out = self.output_norm(x_out)
            
            # Apply self-attention to better capture context dependencies
            attn_output, _ = self.attention(x_out, x_out, x_out)
            
            # Get the last hidden state for prediction
            if self.bidirectional:
                # For bidirectional, concatenate the last hidden states from both directions
                last_hidden = torch.cat([hidden_states[-2], hidden_states[-1]], dim=1)
            else:
                # For unidirectional, use the last hidden state
                last_hidden = hidden_states[-1]
                
        elif self.rnn_type == "transformer":
            # Add positional encoding
            x_proj = self.pos_encoder(x_proj)
            
            # Create mask for transformer (optional)
            if src_key_padding_mask is None:
                # Create a causal mask to prevent attending to future tokens
                mask = generate_square_subsequent_mask(x_proj.size(1)).to(device)
            else:
                mask = None
                
            # Transformer forward pass
            x_out = self.transformer_encoder(x_proj, mask=mask, src_key_padding_mask=src_key_padding_mask)
            
            # Apply output normalization
            x_out = self.output_norm(x_out)
            
            # Use the last token's representation for prediction
            last_hidden = x_out[:, -1, :]
            
            # No hidden states for transformer
            hidden_states = None
        
        # Project to output
        pred = self.fc1(last_hidden)
        pred = F.relu(pred)
        pred = self.dropout(pred)
        pred = self.fc2(pred)  # (batch, pred_len)
        
        # Reshape to match expected output
        pred = pred.view(batch_size, self.pred_len, 1)
        
        return pred, hidden_states
    
    def forward(
        self,
        eeg: Optional[torch.Tensor] = None,
        envelope_context: Optional[torch.Tensor] = None,
        envelope_future: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model with autoregressive prediction.
        
        Args:
            envelope_context: Input envelope of shape (batch, context_len, 1)
                              The initial context (2s)
            envelope_future: Ground truth future envelope of shape (batch, future_len, 1)
                             Used for teacher forcing during training
            eeg: Input EEG of shape (batch, seq_len, channels) - not used in this model
            src_key_padding_mask: Optional mask for padding tokens (batch, seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing (0.0-1.0)
                                  1.0 = always use ground truth, 0.0 = always use model prediction
            
        Returns:
            Dictionary containing:
                'envelope_pred': Predicted future envelope (batch, future_len, 1)
                'context': The initial context used (batch, context_len, 1)
                'envelope': Full sequence (context + prediction) (batch, context_len + future_len, 1)
        """
        batch_size, context_len, _ = envelope_context.shape
        device = envelope_context.device
        
        # Ensure input has the expected length
        assert context_len == self.context_len, f"Expected context length {self.context_len}, got {context_len}"
        
        # Initialize hidden states
        hidden_states = None
        
        # Initialize the output sequence with the initial context
        current_context = envelope_context.clone()
        
        # Initialize the output predictions
        all_predictions = []
        
        # Calculate number of iterations needed
        num_iterations = (self.future_len + self.step_size - 1) // self.step_size
        
        # Determine if we're in training or evaluation mode
        is_training = self.training and envelope_future is not None
        
        # Autoregressive prediction loop
        for i in range(num_iterations):
            # Perform single step forward pass
            pred, hidden_states = self._single_step_forward(current_context, hidden_states, src_key_padding_mask)
            
            # Store prediction
            all_predictions.append(pred)
            
            # Update context for next iteration
            if is_training and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth for next step
                start_idx = i * self.step_size
                end_idx = min(start_idx + self.pred_len, self.future_len)
                
                if end_idx <= start_idx:
                    break
                    
                # Get the ground truth segment
                ground_truth_segment = envelope_future[:, start_idx:end_idx, :]
                
                # If the segment is shorter than pred_len, pad it
                if ground_truth_segment.shape[1] < self.pred_len:
                    padding = torch.zeros(batch_size, self.pred_len - ground_truth_segment.shape[1], 1, device=device)
                    ground_truth_segment = torch.cat([ground_truth_segment, padding], dim=1)
                
                # Update context by removing oldest samples and adding ground truth
                current_context = torch.cat([
                    current_context[:, self.step_size:, :],
                    ground_truth_segment
                ], dim=1)
            else:
                # No teacher forcing: use model's own prediction
                # Update context by removing oldest samples and adding prediction
                current_context = torch.cat([
                    current_context[:, self.step_size:, :],
                    pred[:, :self.step_size, :]
                ], dim=1)
        
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=1)
        
        # Trim to exact future length needed
        if predictions.shape[1] > self.future_len:
            predictions = predictions[:, :self.future_len, :]
        
        # Combine context and prediction for full sequence
        full_sequence = torch.cat([envelope_context, predictions], dim=1)
        
        return {
            'envelope_pred': predictions,
            'context': envelope_context,
            'envelope': full_sequence
        }

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    
    Args:
        sz: Size of the square mask
        
    Returns:
        Tensor of shape [sz, sz] with lower triangular part filled with 0s and upper with -inf
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create random data with shape (batch_size=1, seq_length=128, feature_dim=1)
    # 128 = 2 seconds at 64 Hz (initial context)
    batch_size, context_length, feature_dim = 1, 128, 1
    future_length = 512  # 8 seconds at 64 Hz
    
    # Create a more realistic envelope signal (sinusoidal with noise)
    t = np.linspace(0, 10, context_length + future_length)  # 10 seconds
    freq = 2  # 2 Hz
    
    # Create a sinusoidal signal with some noise
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 5 * t)
    signal = signal + 0.1 * np.random.randn(context_length + future_length)
    
    # Ensure signal is positive (like an envelope)
    signal = np.abs(signal)
    
    # Split into context and future
    context_signal = signal[:context_length]
    future_signal = signal[context_length:]
    
    # Convert to tensors
    context_tensor = torch.tensor(context_signal, dtype=torch.float32).view(batch_size, context_length, feature_dim)
    future_tensor = torch.tensor(future_signal, dtype=torch.float32).view(batch_size, future_length, feature_dim)
    
    # Test autoregressive model with different model types
    model_types = ["lstm", "gru", "transformer"]
    
    plt.figure(figsize=(15, 10))
    
    for i, model_type in enumerate(model_types):
        # Create model
        model = EnvelopeForecasterAutoregressive(
            in_channel=feature_dim, 
            hidden_dim=128, 
            n_layers=2, 
            rnn_type=model_type,
            context_seconds=2.0,
            prediction_seconds=1.0,
            future_seconds=8.0,
            step_size=0.5
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Forward pass with autoregressive prediction
        with torch.no_grad():
            output = model(
                envelope_context=context_tensor,
                envelope_future=future_tensor,
                teacher_forcing_ratio=0.0  # No teacher forcing for evaluation
            )
        
        # Get predictions
        predictions = output['envelope_pred'].squeeze().numpy()
        
        # Plot in a subplot
        plt.subplot(3, 1, i+1)
        
        # Plot initial context
        plt.plot(np.arange(0, context_length), context_signal, 'b-', label='Initial Context (2s)')
        
        # Plot predicted future
        plt.plot(np.arange(context_length, context_length + len(predictions)), predictions, 'r-', label='Predicted Future (8s)')
        
        # Plot ground truth future
        plt.plot(np.arange(context_length, context_length + future_length), future_signal, 'g-', label='Ground Truth Future (8s)')
        
        plt.axvline(x=context_length, color='k', linestyle='--', label='Prediction Start')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(f'Autoregressive Prediction: {model_type.upper()} Model')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('autoregressive_model_test.png')
    plt.close()
    
    print("\nAutoregressive model test completed. Plot saved as 'autoregressive_model_test.png'")
    
    # Test teacher forcing during training
    print("\nTesting teacher forcing during training:")
    
    # Create model
    model = EnvelopeForecasterAutoregressive(
        in_channel=feature_dim, 
        hidden_dim=128, 
        n_layers=2, 
        rnn_type="lstm",
        context_seconds=2.0,
        prediction_seconds=1.0,
        future_seconds=8.0,
        step_size=0.5
    )
    
    # Set to training mode
    model.train()
    
    # Test with different teacher forcing ratios
    teacher_forcing_ratios = [0.0, 0.5, 1.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, tf_ratio in enumerate(teacher_forcing_ratios):
        # Forward pass with teacher forcing
        output = model(
            envelope_context=context_tensor,
            envelope_future=future_tensor,
            teacher_forcing_ratio=tf_ratio
        )
        
        # Get predictions
        predictions = output['envelope_pred'].squeeze().detach().numpy()
        
        # Plot in a subplot
        plt.subplot(3, 1, i+1)
        
        # Plot initial context
        plt.plot(np.arange(0, context_length), context_signal, 'b-', label='Initial Context (2s)')
        
        # Plot predicted future
        plt.plot(np.arange(context_length, context_length + len(predictions)), predictions, 'r-', label='Predicted Future (8s)')
        
        # Plot ground truth future
        plt.plot(np.arange(context_length, context_length + future_length), future_signal, 'g-', label='Ground Truth Future (8s)')
        
        plt.axvline(x=context_length, color='k', linestyle='--', label='Prediction Start')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(f'Teacher Forcing Ratio: {tf_ratio}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('teacher_forcing_test.png')
    plt.close()
    
    print("Teacher forcing test completed. Plot saved as 'teacher_forcing_test.png'") 