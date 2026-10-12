import torch
import torch.nn as nn
import torch.nn.functional as F
import webrtcvad
import numpy as np
from scipy import signal

class WeightedPearsonLoss(nn.Module):
    """
    Weighted Pearson correlation coefficient loss using WebRTC VAD.
    Loss = 1 - correlation with time-dependent weighting based on speech activity
    """
    def __init__(self, 
                 vad_mode: int = 2,       # WebRTC VAD aggressiveness (0-3)
                 weight_low: float = 1.0,
                 weight_high: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            vad_mode: WebRTC VAD aggressiveness (0-3)
                     0 is least aggressive, 3 is most aggressive
            weight_low: Weight for non-speech regions
            weight_high: Weight for speech regions
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.reduction = reduction
        
        # Constants for resampling
        self.target_sr = 8000   # WebRTC VAD works with 8kHz
        self.source_sr = 64     # Original sample rate
        self.frame_duration = 30  # WebRTC VAD frame duration in ms
        self.samples_per_frame = int(self.target_sr * self.frame_duration / 1000)  # 240 samples for 8kHz
    
    def resample_to_8k(self, signal_64hz):
        """Resample signal from 64Hz to 8kHz using linear interpolation."""
        # Convert to numpy
        signal_np = signal_64hz.cpu().numpy()
        
        # Create time arrays
        t_orig = np.linspace(0, len(signal_np) / self.source_sr, len(signal_np))
        t_new = np.linspace(0, len(signal_np) / self.source_sr, 
                           int(len(signal_np) * self.target_sr / self.source_sr))
        
        # Interpolate
        resampled = np.interp(t_new, t_orig, signal_np)
        
        # Scale to int16 range for VAD
        resampled = (resampled * 32768).astype(np.int16)
        return resampled
    
    def compute_vad_weights(self, audio_8k):
        """Compute VAD decisions for 8kHz audio."""
        # Ensure audio is in correct format (int16)
        if not audio_8k.dtype == np.int16:
            audio_8k = (audio_8k * 32768).astype(np.int16)
        
        # Split into frames
        n_frames = len(audio_8k) // self.samples_per_frame
        frames = np.array_split(audio_8k[:n_frames * self.samples_per_frame], n_frames)
        
        # Get VAD decisions
        vad_decisions = []
        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.target_sr)
                vad_decisions.append(float(is_speech))
            except:
                vad_decisions.append(0.0)  # Mark as non-speech if frame is invalid
        
        return np.array(vad_decisions)
    
    def compute_weights(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent weights using WebRTC VAD.
        
        Args:
            y: Target values (batch_size, seq_len, 1)
            
        Returns:
            torch.Tensor: Weights tensor of same shape as input
        """
        batch_size, seq_len, _ = y.shape
        weights_list = []
        
        # Process each sample independently
        for i in range(batch_size):
            # Get single sample
            y_sample = y[i, :, 0]  # (seq_len,)
            
            # Resample to 8kHz
            y_8k = self.resample_to_8k(y_sample)
            
            # Get VAD decisions
            vad_decisions = self.compute_vad_weights(y_8k)
            
            # Interpolate VAD decisions back to original length
            t_vad = np.linspace(0, 1, len(vad_decisions))
            t_orig = np.linspace(0, 1, seq_len)
            vad_decisions_interp = np.interp(t_orig, t_vad, vad_decisions)
            
            # Apply smoothing to avoid abrupt transitions
            window_size = max(3, int(0.05 * self.source_sr))  # 50ms window
            vad_smooth = signal.savgol_filter(vad_decisions_interp, window_size, 2)
            
            # Convert to weights
            weights = torch.tensor(vad_smooth, device=y.device, dtype=y.dtype)
            weights = self.weight_low + (self.weight_high - self.weight_low) * weights
            
            weights_list.append(weights)
        
        # Stack all samples back together
        weights = torch.stack(weights_list, dim=0)  # (batch_size, seq_len)
        return weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted Pearson correlation loss.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            
        Returns:
            torch.Tensor: 1 - weighted_correlation
        """
        # Flatten the tensors to (batch_size, seq_len)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        # Compute weights using WebRTC VAD
        weights = self.compute_weights(y.unsqueeze(-1))
        weights = weights.squeeze(-1)
        
        # Center the variables with weighted mean
        x = x - (x * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        y = y - (y * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        
        # Compute weighted correlation for each sample in batch
        numerator = torch.sum(weights * x * y, dim=1, keepdim=True)
        x_std = torch.sqrt(torch.sum(weights * x**2, dim=1, keepdim=True) + 1e-8)
        y_std = torch.sqrt(torch.sum(weights * y**2, dim=1, keepdim=True) + 1e-8)
        corr = numerator / (x_std * y_std)
        
        # Convert to loss (1 - correlation)
        loss = 1 - corr
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 