import os
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.data.data_utils import get_dataloaders
from src.train.trainer import EnvelopeTrainer
from src.models.model_factory import ModelFactory
from src.models import registry  # This will register all models
from src.train.loss_factory import LossFactory

from test_model import test_model, create_subject_plots
from fine_tune_subjects import fine_tune_subject, create_comparison_plots
from train_model import train_subject_independent, setup_directories

def fine_tune_subjects(
    model_name: str,
    base_checkpoint: str,
    config: Any,
    save_dir: Path,
    subjects: list = None
) -> None:
    """
    Fine-tune the model for each subject.
    
    Args:
        model_name: Name of the model to use
        base_checkpoint: Path to the best subject-independent model
        config: Configuration object
        save_dir: Directory to save fine-tuned models
        subjects: List of subjects to fine-tune for. If None, fine-tunes for all subjects.
    """
    print("\n" + "="*80)
    print(f"Starting Subject-Specific Fine-Tuning for {model_name}")
    print("="*80)
    
    # Get list of subjects if not provided
    if subjects is None:
        _, _, test_loader = get_dataloaders(config)
        subjects = sorted(pd.unique(test_loader.dataset.df['sub']))
    
    # First, test base model on all subjects
    print("\nTesting base model performance...")
    original_subject_id = config.subject_id  # Store original subject_id
    config.subject_id = None  # Test on all subjects
    base_results = test_model(base_checkpoint, config)
    if 'sub' in base_results.columns:
        base_results['subject_id'] = base_results['sub']
    config.subject_id = original_subject_id  # Restore original subject_id
    
    # Store results for all subjects
    fine_tuned_results = []
    
    # Fine-tune for each subject
    for subject_id in subjects:
        print(f"\n{'='*80}")
        print(f"Fine-tuning {model_name} for subject: {subject_id}")
        print(f"{'='*80}")
        
        try:
            # Fine-tune model
            results = fine_tune_subject(
                base_model_checkpoint=base_checkpoint,
                subject_id=subject_id,
                config=config,
                save_dir=save_dir / f"subject_{subject_id}"
            )
            
            # Store results
            if isinstance(results, pd.DataFrame):
                results['subject_id'] = subject_id
                fine_tuned_results.append(results)
            
            # Cleanup after each subject's fine-tuning
            config.cleanup_wandb()
            
        except Exception as e:
            print(f"Error fine-tuning subject {subject_id}: {str(e)}")
            # Cleanup even if there was an error
            config.cleanup_wandb()
            continue
    
    # Save and analyze results
    if fine_tuned_results:
        save_fine_tuning_results(base_results, fine_tuned_results, save_dir)
    
    # Final cleanup after all fine-tuning
    config.cleanup_wandb()

def save_fine_tuning_results(
    base_results: pd.DataFrame,
    fine_tuned_results: list,
    save_dir: Path
) -> None:
    """Save and analyze fine-tuning results."""
    # Combine all results
    all_results = pd.concat(fine_tuned_results)
    
    # Save results
    base_results_path = save_dir / "base_model_results.csv"
    summary_path = save_dir / "fine_tuning_summary.csv"
    base_results.to_csv(base_results_path, index=False)
    all_results.to_csv(summary_path, index=False)
    
    # Create comparison plots
    violin_path, comparison_path = create_comparison_plots(
        base_results,
        all_results,
        save_dir
    )
    
    # Calculate and save statistics
    subject_means = all_results.groupby('subject_id')['correlation'].mean()
    mean_of_means = subject_means.mean()
    std_of_means = subject_means.std()
    sem = std_of_means / np.sqrt(len(subject_means))
    
    # Save detailed summary
    summary_txt_path = save_dir / "fine_tuning_analysis.txt"
    with open(summary_txt_path, 'w') as f:
        f.write("Fine-tuning Analysis Summary:\n")
        f.write("\nBase Model Performance:\n")
        f.write(f"Average Correlation: {base_results['correlation'].mean():.4f}\n")
        f.write(f"Correlation Std: {base_results['correlation'].std():.4f}\n")
        f.write(f"Average Correlation SEM: {base_results['correlation'].std() / np.sqrt(len(base_results)):.4f}\n")
        
        f.write("\nFine-tuned Model Performance:\n")
        f.write(f"Average Correlation: {all_results['correlation'].mean():.4f}\n")
        f.write(f"Correlation Std: {all_results['correlation'].std():.4f}\n")
        f.write(f"Average Correlation SEM: {all_results['correlation'].std() / np.sqrt(len(all_results)):.4f}\n")
        f.write(f"Mean of Subject Means: {mean_of_means:.4f}\n")
        f.write(f"STD of Subject Means: {std_of_means:.4f}\n")
        f.write(f"SEM of Subject Means: {sem:.4f}\n")
        
        f.write("\nImprovement Analysis:\n")
        base_means = base_results.groupby('subject_id')['correlation'].mean()
        ft_means = all_results.groupby('subject_id')['correlation'].mean()
        improvements = ft_means - base_means
        f.write(f"Average Improvement: {improvements.mean():.4f} ± {improvements.std():.4f}\n")
        f.write(f"Number of Improved Subjects: {(improvements > 0).sum()}/{len(improvements)}\n")
    
    # Print summary
    print(f"\nResults saved to: {summary_path}")
    print(f"Analysis saved to: {summary_txt_path}")
    print(f"Comparison plots saved as:")
    print(f"- {violin_path}")
    print(f"- {comparison_path}")
    
    # Print key findings
    print("\nKey Findings:")
    print(f"Base Model Average: {base_results['correlation'].mean():.4f} ± {base_results['correlation'].std()/np.sqrt(len(base_results)):.4f}")
    print(f"Fine-tuned Average: {mean_of_means:.4f} ± {sem:.4f}")
    print(f"Average Improvement: {improvements.mean():.4f} ± {improvements.std():.4f}")
    print(f"Subjects Improved: {(improvements > 0).sum()}/{len(improvements)}")

def main():
    parser = argparse.ArgumentParser(description="Train and fine-tune envelope decoding model")
    
    # Get available models for help text
    available_models = ModelFactory.list_available_models()
    parser.add_argument("--model", type=str, default="baseline_decoder",
                      help=f"Model to use. Available models: {available_models}")
    parser.add_argument("--subjects", type=str, default=None,
                      help="Comma-separated list of subject IDs to fine-tune (e.g., 'sub-001,sub-002')")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip subject-independent training and use existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to existing checkpoint (required if --skip-training is used)")
    parser.add_argument("--loss-function", type=str, default="mse",
                      help=f"Loss function to use. Available losses: {LossFactory.list_available_losses()}")
    parser.add_argument("--test-mode", action="store_true",
                      help="Run in test mode with reduced epochs and single subject")
    
    args = parser.parse_args()
    
    # Validate model name
    if args.model not in available_models:
        raise ValueError(f"Unknown model: {args.model}. Available models: {available_models}")
    
    # Get the appropriate config class and create instance
    ConfigClass = ModelFactory.get_config_class(args.model)
    config = ConfigClass()
    config.model_name = args.model
    config.loss_function = args.loss_function
    
    # Modify config for test mode
    if args.test_mode:
        print("\nRunning in TEST MODE with reduced epochs")
        # Reduce epochs for faster testing
        config.num_epochs = 100  # Just 2 epochs for testing
        config.finetune_epochs = 20
        config.batch_size = 32  # Smaller batch size
        config.patience = 5  # Reduce patience for early stopping
        print(f"Test mode settings: epochs={config.num_epochs}, "
              f"finetune_epochs={config.finetune_epochs}, "
              f"batch_size={config.batch_size}")
    
    # Set up directory structure
    directories = setup_directories(args.model, args.loss_function)
    
    # Get base model checkpoint
    if args.skip_training:
        if args.checkpoint is None:
            raise ValueError("Must provide --checkpoint when using --skip-training")
        base_checkpoint = args.checkpoint
    else:
        # Train subject-independent model
        base_checkpoint = train_subject_independent(
            model_name=args.model,
            config=config,
            save_dir=directories['subject_independent']
        )
    
    # Determine subjects to fine-tune
    subjects = None
    if args.subjects:
        subjects = args.subjects.split(',')
        print(f"\nFine-tuning for specific subjects: {subjects}")
    
    # Fine-tune for each subject
    fine_tune_subjects(
        model_name=args.model,
        base_checkpoint=base_checkpoint,
        config=config,
        save_dir=directories['fine_tuned'],
        subjects=subjects
    )

if __name__ == "__main__":
    main() 