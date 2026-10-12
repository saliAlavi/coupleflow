from dataclasses import dataclass
from typing import Optional
import wandb
import gc
import os
import shutil
import torch

@dataclass
class BaselineConfig:
    """Configuration for baseline envelope decoder training."""
    
    # Model Configuration
    n_channels: int = 64
    filter_length_ms: float = 500
    sampling_rate: int = 64
    dropout: float = 0.2
    predict_mask: bool = False  # Whether to predict activity mask along with envelope
    
    # Envelope Processing Configuration
    envelope_filter_type: str = 'none'  # Options: none, lowpass_4hz, lowpass_8hz, bandpass_1_4hz, bandpass_4_8hz, highpass_1hz
    envelope_sampling_rate: float = 64.0  # Sampling rate for envelope filtering
    
    # Training Configuration
    seed: int = 42  # Random seed for reproducibility
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss_function: str = "mse"  # Options: "mse", "l1", "smooth_l1"
    
    # Data Configuration
    train_filelist: str = "filelists/filelist_5s_80pct/train_80pct_5s.csv"
    val_filelist: str = "filelists/filelist_5s_80pct/val_80pct_5s.csv"
    test_filelist: str = "filelists/filelist_5s_80pct_test_no_overlap/test_no_overlap_5s.csv"
    data_base_dir: str = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    
    # Data Augmentation Configuration
    eeg_time_mask_percent: float = 0.0  # Percentage of time steps to mask in EEG (0-1)
    eeg_channel_mask_percent: float = 0.0  # Percentage of channels to mask in EEG (0-1)
    envelope_time_mask_percent: float = 0.0  # Percentage of time steps to mask in envelope (0-1)
    mask_prob: float = 0.5  # Probability of applying masking to each sample (0-1)

    # Contextual Configuration
    is_contextual: bool = False  # Flag for contextual models (e.g., inpainting models)
    
    # Forecasting Configuration
    context_seconds: float = 1.5   # Number of seconds of context (fixed at 2 seconds)
    prediction_seconds: float = 3.5  # Number of seconds to predict (fixed at 1 second)
    sampling_rate: int = 64        # Sampling rate in Hz
    context_mask_strategy: str = "none"  # Options: "deterministic", "random", "none"
    mask_value: float = 0.0  # Value to mask the context with
    
    # Trainer Configuration
    device: str = "cuda:0"  # or "cpu"
    log_interval: int = 50  # Steps between logging
    patience: int = 5  # Early stopping patience
    checkpoint_dir: str = "checkpoints_5s"
    
    # Subject Configuration
    subject_id: Optional[str] = None  # For subject-dependent training
    
    # Experiment Configuration
    experiment_name: str = "baseline_model"
    wandb_project: str = "eeg_decoding_5s_window"
    
    # Fine-tuning Configuration
    finetune_epochs: int = 50
    finetune_learning_rate: float = 1e-4
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @staticmethod
    def cleanup_wandb():
        """Cleanup wandb runs and temporary files."""
        try:
            # Finish any active wandb run
            if wandb.run is not None:
                try:
                    wandb.finish()
                except:
                    pass
            
            # Clean up wandb files
            if os.path.exists("wandb"):
                # Remove debug logs and temporary files
                for f in os.listdir("wandb"):
                    if any(f.startswith(prefix) for prefix in ["debug", "latest-run", "run-", "tmp"]):
                        try:
                            path = os.path.join("wandb", f)
                            if os.path.isfile(path):
                                os.remove(path)
                            elif os.path.isdir(path):
                                shutil.rmtree(path)
                        except Exception as e:
                            print(f"Warning: Could not remove {f}: {str(e)}")
                
                # Remove empty run directories
                for d in os.listdir("wandb"):
                    dir_path = os.path.join("wandb", d)
                    try:
                        if os.path.isdir(dir_path):
                            # Check if directory is empty or only contains empty subdirectories
                            if not any(os.path.exists(os.path.join(root, f)) 
                                     for root, _, files in os.walk(dir_path) 
                                     for f in files):
                                shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"Warning: Could not process directory {d}: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
        except Exception as e:
            print(f"Warning: Error during wandb cleanup: {str(e)}")
            # Continue even if cleanup fails 