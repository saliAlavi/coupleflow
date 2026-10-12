import os
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any

from src.models.model_factory import ModelFactory
from src.train.loss_factory import LossFactory
from src.models import registry  # This will register all models
from src.data.data_utils import get_dataloaders
from src.train.trainer import EnvelopeTrainer
from test_model import test_model, create_subject_plots

def setup_directories(model_name: str, loss_name: str, config: Any, filter_type: str = 'none') -> Dict[str, Path]:
    """
    Create directory structure for the experiment.
    
    Args:
        model_name: Name of the model (e.g., 'baseline_decoder')
        loss_name: Name of the loss function (e.g., 'mse', 'pearson')
        config: Configuration object containing masking parameters
        filter_type: Type of envelope filter (e.g., 'none', 'lowpass_4hz')
        
    Returns:
        Dictionary containing paths to important directories
    """
    # Set up directory structure
    base_dir = Path(config.checkpoint_dir)
    if filter_type != 'none':
        base_dir = Path(f"checkpoints_{filter_type}")
    
    # Convert model name to title case for directory naming
    model_dir_name = "".join(word.title() for word in model_name.split('_'))
    
    # Get masking configuration from config
    time_mask = getattr(config, 'eeg_time_mask_percent', 0.0)
    channel_mask = getattr(config, 'eeg_channel_mask_percent', 0.0)
    
    # Add masking info to directory name if any masking is applied
    masking_str = ""
    if time_mask > 0 or channel_mask > 0:
        masking_str = f"_tm{time_mask:.1f}_cm{channel_mask:.1f}"
    
    model_dir = base_dir / f"{model_dir_name}_{loss_name}{masking_str}"
    subject_independent_dir = model_dir / "subject_independent"
    fine_tuned_dir = model_dir / "subject_dependent_fine_tuned"
    
    # Create directories
    for directory in [subject_independent_dir, fine_tuned_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        "model": model_dir,
        "subject_independent": subject_independent_dir,
        "fine_tuned": fine_tuned_dir
    }

def train_subject_independent(model_name: str, config: Any, save_dir: Path) -> str:
    """
    Train a subject-independent model.
    
    Args:
        model_name: Name of the model to train
        config: Configuration object
        save_dir: Directory to save checkpoints
        
    Returns:
        Path to best checkpoint
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize model using factory
    model = ModelFactory.create(model_name, config)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = LossFactory.create(config.loss_function)
    print(f"Using loss function: {config.loss_function}")
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.experiment_name}_{timestamp}"
    
    # Create checkpoint directory
    checkpoint_dir = save_dir / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer with wandb tags
    trainer = EnvelopeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config.to_dict(),
        experiment_name=experiment_name,
        checkpoint_dir=str(checkpoint_dir),
        wandb_project=config.wandb_project,
        wandb_tags=[
            f"model:{model_name}",
            f"loss:{config.loss_function}",
            "phase:subject_independent"
        ]
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train(config.num_epochs)
    
    # Return path to best model
    best_model_path = checkpoint_dir / "best_model.pt"
    return str(best_model_path)

def main():
    parser = argparse.ArgumentParser(description="Train a subject-independent envelope decoding model")
    # Base arguments
    parser.add_argument("--model", type=str, default="BaselineDecoder",
                      help=f"Model to use. Available models: {list(ModelFactory._models.keys())}")
    parser.add_argument("--loss_function", type=str, default="mse",
                      help=f"Loss function to use. Options: {list(LossFactory._losses.keys())}")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip training and use existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to existing checkpoint (required if --skip-training is used)")
    parser.add_argument("--test-mode", action="store_true",
                      help="Run in test mode with reduced epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                      help="Directory to save checkpoints")
    # Envelope filter arguments
    parser.add_argument("--envelope_filter_type", type=str, default="none",
                      help="Type of filter to apply to envelope. Options: none, lowpass_4hz, lowpass_8hz, bandpass_1_4hz, bandpass_4_8hz, highpass_1hz")
    parser.add_argument("--envelope_sampling_rate", type=float, default=64.0,
                      help="Sampling rate for envelope filtering in Hz")
    
    # Model architecture arguments
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_inner", type=int, help="Inner dimension of feed-forward network")
    parser.add_argument("--n_head", type=int, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--in_channel", type=int, help="Input channel dimension")
    parser.add_argument("--seq_len", type=int, help="Sequence length")
    
    # Position embedding and attention arguments
    parser.add_argument("--pe_method", type=str, help="Position embedding method")
    parser.add_argument("--rope_mode", type=str, help="RoPE mode")
    parser.add_argument("--qkv_bias", type=lambda x: x.lower() == 'true', help="Whether to use bias in QKV projection")
    parser.add_argument("--qk_norm", type=str, help="QK normalization method")
    parser.add_argument("--norm_layer", type=str, help="Normalization layer type")
    
    # Training arguments
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--use_checkpoint", type=lambda x: x.lower() == 'true', help="Whether to use checkpointing")
    
    # Data augmentation arguments
    parser.add_argument("--eeg_time_mask_percent", type=float, help="Percentage of time steps to mask in EEG (0-1)")
    parser.add_argument("--eeg_channel_mask_percent", type=float, help="Percentage of channels to mask in EEG (0-1)")
    parser.add_argument("--envelope_time_mask_percent", type=float, help="Percentage of time steps to mask in envelope (0-1)")
    
    parser.add_argument("--within_sub_num", type=int, help="Within subject number")
    
    args = parser.parse_args()
    
    # Get the appropriate config class and create instance
    print(f"Getting config class from model {args.model}")
    ConfigClass = ModelFactory.get_config_class(args.model)  # Get config class from factory
    config = ConfigClass()
    config.model_name = args.model
    config.loss_function = args.loss_function
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None:  # Only update if argument was provided
            setattr(config, key, value)
    
    # Modify config for test mode
    if args.test_mode:
        print("\nRunning in TEST MODE with reduced epochs")
        config.num_epochs = 1  # Just 2 epochs for testing
        config.patience = 1  # Reduce patience for early stopping
        config.train_filelist = config.val_filelist  # Use lite training set
        config.checkpoint_dir = "checkpoints_test"
        print(f"Test mode settings: epochs={config.num_epochs}, "
              f"batch_size={config.batch_size}, "
              f"train_filelist={config.train_filelist}")
    
    # Setup directory structure with filter type
    directories = setup_directories(args.model, args.loss_function, config, args.envelope_filter_type)
    
    # Get model checkpoint
    if args.skip_training:
        if args.checkpoint is None:
            raise ValueError("Must provide --checkpoint when using --skip-training")
        best_model_path = args.checkpoint
    else:
        # Train subject-independent model
        best_model_path = train_subject_independent(args.model, config, directories['subject_independent'])
    
    print("\nTraining completed!")
    print(f"Best model saved at: {best_model_path}")
    
    # Test the model
    print("\nTesting model performance...")
    test_results = test_model(best_model_path, config)
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main() 