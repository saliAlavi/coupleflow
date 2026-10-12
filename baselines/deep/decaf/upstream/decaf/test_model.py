import os
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.model_factory import ModelFactory
from src.models import registry  # This will register all models
from src.data.data_utils import get_dataloaders
from src.train.trainer import EnvelopeTrainer
from configs.baseline_config import BaselineConfig

def create_subject_plots(results_df, save_dir: Path, timestamp: str):
    """
    Create and save violin and box plots for per-subject correlations.
    
    Args:
        results_df: DataFrame containing test results
        save_dir: Directory to save plots
        timestamp: Timestamp string for file naming
    """
    # Calculate mean correlation per subject for sorting
    subject_means = results_df.groupby('sub')['correlation'].mean().sort_values(ascending=True)
    sorted_subjects = subject_means.index
    
    # Create figure for single violin plot of subject means
    plt.figure(figsize=(10, 6))
    # Create a DataFrame with just the mean correlations
    means_df = pd.DataFrame({'Mean Correlation': subject_means.values})
    sns.violinplot(data=means_df, y='Mean Correlation')
    plt.ylabel('Mean Pearson Correlation')
    plt.title('Distribution of Mean Subject Correlations')
    plt.tight_layout()
    violin_path = save_dir / f"subject_means_violin_{timestamp}.png"
    plt.savefig(violin_path)
    plt.close()
    
    # Create figure for box plot
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=results_df, x='sub', y='correlation', order=sorted_subjects)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Subject')
    plt.ylabel('Pearson Correlation')
    plt.title('Distribution of Correlations by Subject (Box Plot)')
    plt.tight_layout()
    box_path = save_dir / f"subject_correlations_box_{timestamp}.png"
    plt.savefig(box_path)
    plt.close()
    
    # Create summary plot with mean correlations
    plt.figure(figsize=(15, 8))
    subject_means.plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Subject')
    plt.ylabel('Mean Pearson Correlation')
    plt.title('Mean Correlation by Subject')
    plt.tight_layout()
    means_path = save_dir / f"subject_correlations_means_{timestamp}.png"
    plt.savefig(means_path)
    plt.close()
    
    # Print summary statistics
    print("\nPer-Subject Statistics:")
    print(f"Best performing subject: {subject_means.index[-1]} (r={subject_means.iloc[-1]:.4f})")
    print(f"Worst performing subject: {subject_means.index[0]} (r={subject_means.iloc[0]:.4f})")
    print(f"Median performance: r={subject_means.median():.4f}")
    
    return violin_path, box_path, means_path

def test_model(checkpoint_path: str, config, teacher_forcing: bool = False):
    """Test a trained model."""
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set up device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create dataloaders (we only need test loader but get_dataloaders returns all three)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize model using factory
    model = ModelFactory.create(config.model_name, config)
    print(f"Created model of type: {type(model).__name__}")
    model = model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint model state dict keys: {checkpoint['model_state_dict'].keys()}")
    print(f"Current model state dict keys: {model.state_dict().keys()}")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Initialize trainer for testing
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())  # Not used during testing
    
    # Get checkpoint directory and create timestamp
    chkpt_dir = Path(checkpoint_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"test_{config.experiment_name}_{timestamp}"
    
    # Create results directory using checkpoint name and timestamp
    chkpt_name = Path(checkpoint_path).stem  # gets filename without extension
    if teacher_forcing:
        results_dir = chkpt_dir / f"{chkpt_name}_test_results_teacher_forcing_{timestamp}"
    else:
        results_dir = chkpt_dir / f"{chkpt_name}_test_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results in: {results_dir}")
    
    # Initialize trainer
    trainer = EnvelopeTrainer(
        model=model,
        train_loader=train_loader, 
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config.to_dict(),
        experiment_name=experiment_name,
        checkpoint_dir=str(results_dir),  # Save checkpoints in results dir
        wandb_project=config.wandb_project,
        wandb_tags=[
            f"model:{config.model_name}",
            f"loss:{config.loss_function}",
            "phase:testing",
            f"checkpoint:{Path(checkpoint_path).stem}"
        ]
    )
    
    # Add test mode tags if in test mode
    if hasattr(config, 'test_mode') and config.test_mode:
        trainer.wandb_tags.extend([
            "mode:test",
            f"test_batch_size:{config.batch_size}"
        ])
        
    # Add subject-specific tag if testing on specific subject
    if config.subject_id is not None:
        trainer.wandb_tags.append(f"subject:{config.subject_id}")
    
    # Set test loader and run evaluation
    trainer.test_loader = test_loader
    test_results = trainer.test(teacher_forcing=teacher_forcing)
    
    # Save results with timestamp
    results_path = results_dir / "test_results.csv"
    test_results.to_csv(results_path, index=False)
    print(f"\nTest results saved to: {results_path}")
    
    # Create and save subject-wise plots
    violin_path, box_path, means_path = create_subject_plots(test_results, results_dir, timestamp)
    print(f"\nPlots saved as:")
    print(f"Violin plot: {violin_path}")
    print(f"Box plot: {box_path}")
    print(f"Means plot: {means_path}")
    
    # Save summary statistics to a text file
    summary_path = results_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Test Results Summary:\n")
        f.write(f"Average Correlation: {test_results['correlation'].mean():.4f}\n")
        f.write(f"Correlation Std: {test_results['correlation'].std():.4f}\n")
        f.write(f"Average Correlation SEM: {test_results['correlation'].std() / np.sqrt(len(test_results)):.4f}\n")
        
        # Calculate and write subject means statistics
        subject_means = test_results.groupby('sub')['correlation'].mean()
        mean_of_means = subject_means.mean()
        std_of_means = subject_means.std()
        sem = std_of_means / np.sqrt(len(subject_means))
        f.write(f"Mean of Subject Means: {mean_of_means:.4f}\n")
        f.write(f"STD of Subject Means: {std_of_means:.4f}\n")
        f.write(f"SEM (Standard Error of Subject Means): {sem:.4f}\n")
        
        f.write(f"Average MSE: {test_results['mse'].mean():.4f}\n")
        f.write(f"MSE Std: {test_results['mse'].std():.4f}\n")
        f.write(f"Number of subjects: {test_results['sub'].nunique()}\n")
        f.write(f"Number of audio files: {test_results['audio_file'].nunique()}\n\n")
        
        f.write("Per-Subject Statistics:\n")
        subject_means = test_results.groupby('sub')['correlation'].mean().sort_values(ascending=True)
        f.write(f"Best performing subject: {subject_means.index[-1]} (r={subject_means.iloc[-1]:.4f})\n")
        f.write(f"Worst performing subject: {subject_means.index[0]} (r={subject_means.iloc[0]:.4f})\n")
        f.write(f"Median performance: r={subject_means.median():.4f}\n")
    
    # Print summary statistics
    print("\nTest Results Summary:")
    print(f"Average Correlation: {test_results['correlation'].mean():.4f}")
    print(f"Correlation Std: {test_results['correlation'].std():.4f}")
    print(f"Average Correlation SEM: {test_results['correlation'].std() / np.sqrt(len(test_results)):.4f}")
    print(f"Mean of Subject Means: {mean_of_means:.4f}")
    print(f"STD of Subject Means: {std_of_means:.4f}")
    print(f"SEM (Standard Error of Subject Means): {sem:.4f}")
    print(f"Average MSE: {test_results['mse'].mean():.4f}")
    print(f"MSE Std: {test_results['mse'].std():.4f}")
    print(f"Number of subjects: {test_results['sub'].nunique()}")
    print(f"Number of audio files: {test_results['audio_file'].nunique()}")
    
    return test_results

def main():
    parser = argparse.ArgumentParser(description="Test a trained envelope decoding model")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="BaselineDecoder",
                      help=f"Model to use. Available models: {ModelFactory.list_available_models()}")
    parser.add_argument("--teacher_forcing", type=bool, default=False,
                      help="Whether to use teacher forcing")
    args = parser.parse_args()
    
    # Get the appropriate config class and create instance
    ConfigClass = ModelFactory.get_config_class(args.model)
    config = ConfigClass()
    config.model_name = args.model  # Set model name in config
    
    # Test model
    test_results = test_model(args.checkpoint, config, args.teacher_forcing)

if __name__ == "__main__":
    main() 