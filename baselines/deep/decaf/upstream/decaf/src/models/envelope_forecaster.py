import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math
import numpy as np

class EnvelopeForecaster(nn.Module):
    """
    Recurrent model for envelope forecasting.
    Takes 2 seconds of context and predicts the next 1 second.
    
    With a sampling rate of 64 Hz:
    - 2 seconds context = 128 samples
    - 1 second prediction = 64 samples
    
    Special case: When context_seconds is 0, the model will generate a random
    low-pass filtered signal instead of using any model.
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
        nhead: int = 4,  # Number of attention heads for transformer
        **kwargs
    ):
        super().__init__()
        
        # Save parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.sr = 64  # Sampling rate
        self.context_seconds = context_seconds
        self.prediction_seconds = prediction_seconds
        
        # Calculate context and prediction lengths
        self.context_len = int(context_seconds * self.sr)  # 2 seconds at 64 Hz = 128 samples
        self.pred_len = int(prediction_seconds * self.sr)     # 1 second at 64 Hz = 64 samples
        self.total_len = self.context_len + self.pred_len  # 3 seconds total
        
        # Skip model initialization if context_seconds is 0
        if context_seconds == 0:
            return
        
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
        
        # Store the last hidden features for fusion
        self._last_hidden_features = None
    
    def generate_random_lowpass_signal(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        """
        Generate a random low-pass filtered signal.
        
        Args:
            batch_size: Number of signals to generate
            length: Length of each signal
            device: Device to create the tensor on
            
        Returns:
            Tensor of shape (batch_size, length, 1) containing random low-pass filtered signals
        """
        # Generate random noise
        noise = torch.randn(batch_size, length, 1, device=device)
        
        # Apply low-pass filtering using a simple moving average
        window_size = 10  # Adjust this for different cutoff frequencies
        kernel = torch.ones(1, 1, window_size, device=device) / window_size
        
        # Reshape for conv1d (batch, channels, length)
        noise_reshaped = noise.transpose(1, 2)
        
        # Apply convolution for low-pass filtering
        filtered = F.conv1d(noise_reshaped, kernel, padding=window_size//2)
        
        # Reshape back to original shape
        filtered = filtered.transpose(1, 2)
        
        # Scale to reasonable amplitude range for an envelope (positive values)
        filtered = torch.abs(filtered) * 0.5

        # make sure the filtered signal is zero mean and unit variance
        filtered = (filtered - filtered.mean(dim=1, keepdim=True)) / filtered.std(dim=1, keepdim=True)

        # make sure the shape is the same as pred_len
        filtered = filtered[:, :self.pred_len, :]
        
        return filtered
        
    def get_features(self) -> torch.Tensor:
        """
        Returns the last hidden features for use in fusion strategies.
        
        Returns:
            Tensor containing the last hidden features
        """
        if self._last_hidden_features is None:
            raise ValueError("No features available. Call forward() first.")
        return self._last_hidden_features
    
    def forward(
        self,
        eeg: Optional[torch.Tensor] = None,
        envelope_context: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            envelope_context: Input envelope of shape (batch, seq_len_context, 1)
            eeg: Input EEG of shape (batch, seq_len_prediction, channels) - not used in this model
            src_key_padding_mask: Optional mask for padding tokens (batch, seq_len)
            
        Returns:
            Dictionary containing:
                'envelope_pred': Predicted next 1 second of envelope (batch, pred_len, 1)
                'context': The 2-second context used (batch, context_len, 1)
                'envelope': Full 3-second sequence (input + prediction) (batch, total_len, 1)
        """
        batch_size, seq_len, _ = envelope_context.shape
        device = envelope_context.device
        
        # Special case: If context_seconds is 0, generate random low-pass signal
        if self.context_seconds == 0:
            # Generate random prediction
            pred = self.generate_random_lowpass_signal(batch_size, self.pred_len, device)
            
            # Create empty context (just for API compatibility)
            context = torch.zeros(batch_size, self.context_len, 1, device=device)
            
            # Combine context and prediction for full sequence
            full_sequence = torch.cat([context, pred], dim=1)
            
            # Create random features for fusion
            self._last_hidden_features = torch.randn(batch_size, self.hidden_dim, device=device)
            
            return {
                'envelope_pred': pred,
                'context': context,
                'envelope': pred
            }
        
        # Normal case: Use the model
        # Ensure input has the expected length
        assert seq_len == self.context_len, f"Expected input length {self.context_len}, got {seq_len}"
        
        # Split into context and target
        context = envelope_context[:, :self.context_len, :]
        
        # Project input to hidden dimension
        x = self.input_proj(context.transpose(1, 2))  # (batch, hidden_dim, context_len)
        x = x.transpose(1, 2)  # (batch, context_len, hidden_dim)
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Process based on model type
        if self.rnn_type == "lstm":
            # Initialize hidden states
            h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                            batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                            batch_size, self.hidden_dim, device=device)
            # LSTM forward pass
            x, (h_n, _) = self.rnn(x, (h0, c0))
            
            # Apply output normalization
            x = self.output_norm(x)
            
            # Apply self-attention to better capture context dependencies
            attn_output, _ = self.attention(x, x, x)
            
            # Get the last hidden state for prediction
            if self.bidirectional:
                # For bidirectional, concatenate the last hidden states from both directions
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                # For unidirectional, use the last hidden state
                last_hidden = h_n[-1]
                
        elif self.rnn_type == "gru":
            # Initialize hidden states
            h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), 
                            batch_size, self.hidden_dim, device=device)
            # GRU forward pass
            x, h_n = self.rnn(x, h0)
            
            # Apply output normalization
            x = self.output_norm(x)
            
            # Apply self-attention to better capture context dependencies
            attn_output, _ = self.attention(x, x, x)
            
            # Get the last hidden state for prediction
            if self.bidirectional:
                # For bidirectional, concatenate the last hidden states from both directions
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                # For unidirectional, use the last hidden state
                last_hidden = h_n[-1]
                
        elif self.rnn_type == "transformer":
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Create mask for transformer (optional)
            if src_key_padding_mask is None:
                # Create a causal mask to prevent attending to future tokens
                mask = generate_square_subsequent_mask(x.size(1)).to(device)
            else:
                mask = None
                
            # Transformer forward pass
            x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
            
            # Apply output normalization
            x = self.output_norm(x)
            
            # Use the last token's representation for prediction
            last_hidden = x[:, -1, :]
        
        # Store the last hidden features for fusion
        self._last_hidden_features = last_hidden
        
        # Project to output
        pred = self.fc1(last_hidden)
        pred = F.relu(pred)
        pred = self.dropout(pred)
        pred = self.fc2(pred)  # (batch, pred_len)
        
        # Reshape to match expected output
        pred = pred.view(batch_size, self.pred_len, 1)
        
        # Combine context and prediction for full sequence
        full_sequence = torch.cat([context, pred], dim=1)
        
        return {
            'envelope_pred': pred,
            'context': context,
            'envelope': full_sequence
        }
    
    def predict_autoregressive(
        self, 
        initial_context: torch.Tensor, 
        future_seconds: float = 8.0,
        overlap_ratio: float = 0.5,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively predict future envelope values using a sliding window approach.
        This method can be used to generate predictions for the entire sequence.
        
        Args:
            initial_context: Initial context of shape (batch, context_samples, 1)
                            Should be 2 seconds of data (128 samples at 64 Hz)
            future_seconds: Number of seconds to predict into the future
            overlap_ratio: Ratio of overlap between consecutive predictions (0.0-1.0)
                          Higher values provide smoother transitions but may reduce diversity
            temperature: Temperature for sampling (1.0 = no change, <1.0 = more conservative, >1.0 = more diverse)
            
        Returns:
            Tensor of shape (batch, context_samples + future_samples, 1) containing the
            original context followed by the predicted future values
        """
        batch_size, context_samples, channels = initial_context.shape
        device = initial_context.device
        
        # Special case: If context_seconds is 0, generate random low-pass signal for the entire future
        if self.context_seconds == 0:
            # Calculate number of future samples to predict
            future_samples = int(future_seconds * self.sr)
            
            # Generate random prediction for the entire future
            future_signal = self.generate_random_lowpass_signal(batch_size, future_samples, device)
            
            # Combine initial context and future prediction
            output_sequence = torch.cat([initial_context, future_signal], dim=1)
            
            # Create random features for fusion
            self._last_hidden_features = torch.randn(batch_size, self.hidden_dim, device=device)
            
            return output_sequence
        
        # Normal case: Use the model
        # Ensure initial context has the correct length
        assert context_samples == self.context_len, \
            f"Expected initial context length {self.context_len}, got {context_samples}"
        
        # Calculate number of future samples to predict
        future_samples = int(future_seconds * self.sr)
        
        # Calculate step size for sliding window based on overlap ratio
        step_size = max(1, int(self.pred_len * (1 - overlap_ratio)))
        
        # Initialize the output sequence with the initial context
        output_sequence = initial_context.clone()
        
        # Current context starts with the initial context
        current_context = initial_context.clone()
        
        # Calculate number of iterations needed
        num_iterations = (future_samples + step_size - 1) // step_size
        
        for i in range(num_iterations):
            # Prepare input for the model (context + placeholder for prediction)
            model_input = torch.cat([
                current_context,
                torch.zeros(batch_size, self.pred_len, channels, device=device)
            ], dim=1)
            
            # Get model prediction
            output = self(envelope_context=model_input)
            prediction = output['envelope_pred']
            
            # Apply temperature scaling if needed
            if temperature != 1.0:
                # Add some noise scaled by temperature
                noise = torch.randn_like(prediction) * (temperature - 1.0) * 0.1
                prediction = prediction + noise
                # Ensure values are positive (for envelope)
                prediction = torch.abs(prediction)
            
            # Append prediction to output sequence
            if i == 0:
                # For the first iteration, append the full prediction
                output_sequence = torch.cat([output_sequence, prediction], dim=1)
            else:
                # For subsequent iterations, only append the non-overlapping part
                output_sequence = torch.cat([
                    output_sequence, 
                    prediction[:, -step_size:, :]
                ], dim=1)
            
            # Update current context for next iteration
            # Use the last context_len samples as the new context
            current_context = output_sequence[:, -self.context_len:, :]
            
            # If we've predicted enough samples, stop
            if output_sequence.shape[1] >= context_samples + future_samples:
                break
        
        # Trim to exact length needed
        final_length = context_samples + future_samples
        if output_sequence.shape[1] > final_length:
            output_sequence = output_sequence[:, :final_length, :]
        
        # Store the last hidden features for fusion
        # This allows the features to be accessed after autoregressive prediction
        if hasattr(self, '_last_hidden_features') and self._last_hidden_features is not None:
            # We keep the features from the last forward pass
            pass
        
        return output_sequence


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
    
    # Create a more realistic envelope signal (sinusoidal with noise)
    t = np.linspace(0, 2, context_length)  # 2 seconds
    freq = 2  # 2 Hz
    
    # Create a sinusoidal signal with some noise
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 5 * t)
    signal = signal + 0.1 * np.random.randn(context_length)
    
    # Ensure signal is positive (like an envelope)
    signal = np.abs(signal)
    
    # Convert to tensor
    initial_context = torch.tensor(signal, dtype=torch.float32).view(batch_size, context_length, feature_dim)
    
    # Test autoregressive prediction with different model types
    model_types = ["lstm", "gru", "transformer"]
    future_seconds = 8.0  # Predict 8 seconds into the future
    
    plt.figure(figsize=(15, 10))
    
    for i, model_type in enumerate(model_types):
        # Create model
        model = EnvelopeForecaster(
            in_channel=feature_dim, 
            hidden_dim=128, 
            n_layers=2, 
            rnn_type=model_type,
            context_seconds=2.0,
            prediction_seconds=1.0
        )
        
        # Perform autoregressive prediction
        full_sequence = model.predict_autoregressive(
            initial_context=initial_context,
            future_seconds=future_seconds,
            overlap_ratio=0.5,
            temperature=1.0
        )
        
        # Convert to numpy for plotting
        full_sequence_np = full_sequence.squeeze().numpy()
        
        # Plot in a subplot
        plt.subplot(3, 1, i+1)
        
        # Plot initial context
        plt.plot(np.arange(0, context_length), full_sequence_np[:context_length], 'b-', label='Initial Context (2s)')
        
        # Plot predicted future
        plt.plot(np.arange(context_length, len(full_sequence_np)), full_sequence_np[context_length:], 'r-', label='Predicted Future (8s)')
        
        plt.axvline(x=context_length, color='k', linestyle='--', label='Prediction Start')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(f'Autoregressive Prediction: {model_type.upper()} Model')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('autoregressive_prediction.png')
    plt.close()
    
    print("\nAutoregressive prediction test completed. Plot saved as 'autoregressive_prediction.png'")
    
    # Test individual models
    print("\nTesting individual models:")
    
    # Test LSTM model
    lstm_model = EnvelopeForecaster(in_channel=feature_dim, hidden_dim=128, n_layers=2, rnn_type="lstm")
    x = torch.randn(batch_size, 192, feature_dim)  # 3 seconds total (2s context + 1s prediction)
    lstm_output = lstm_model(envelope_context=x)
    print("\nLSTM Model:")
    print("Input shape:", x.shape)                       # (1, 192, 1)
    print("Context shape:", lstm_output['context'].shape)     # (1, 128, 1)
    print("Prediction shape:", lstm_output['envelope_pred'].shape)  # (1, 64, 1)
    print("Full sequence shape:", lstm_output['envelope'].shape)  # (1, 192, 1)
    
    # Test GRU model
    gru_model = EnvelopeForecaster(in_channel=feature_dim, hidden_dim=128, n_layers=2, rnn_type="gru")
    gru_output = gru_model(envelope_context=x)
    print("\nGRU Model:")
    print("Input shape:", x.shape)                       # (1, 192, 1)
    print("Context shape:", gru_output['context'].shape)     # (1, 128, 1)
    print("Prediction shape:", gru_output['envelope_pred'].shape)  # (1, 64, 1)
    print("Full sequence shape:", gru_output['envelope'].shape)  # (1, 192, 1)
    
    # Test Transformer model
    transformer_model = EnvelopeForecaster(in_channel=feature_dim, hidden_dim=128, n_layers=2, rnn_type="transformer")
    transformer_output = transformer_model(envelope_context=x)
    print("\nTransformer Model:")
    print("Input shape:", x.shape)                       # (1, 192, 1)
    print("Context shape:", transformer_output['context'].shape)     # (1, 128, 1)
    print("Prediction shape:", transformer_output['envelope_pred'].shape)  # (1, 64, 1)
    print("Full sequence shape:", transformer_output['envelope'].shape)  # (1, 192, 1) 