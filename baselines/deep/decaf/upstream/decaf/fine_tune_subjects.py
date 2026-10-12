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

from src.models.model_factory import ModelFactory
from src.models import registry  # This will register all models
from src.data.data_utils import get_dataloaders
from src.train.trainer import EnvelopeTrainer
from configs.baseline_config import BaselineConfig
from test_model import test_model, create_subject_plots
from src.train.loss_factory import LossFactory

def fine_tune_subject(
    base_model_checkpoint: str,
    subject_id: str,
    config: BaselineConfig,
    save_dir: Path
) -> Dict[str, Any]:
    """
    Fine-tune the base model for a specific subject.
    If a best model already exists for the subject, skip training and just test.
    
    Args:
        base_model_checkpoint: Path to the base model checkpoint
        subject_id: Subject ID to fine-tune for
        config: Configuration object
        save_dir: Directory to save fine-tuned model and results
    
    Returns:
        Dictionary containing test results
    """
    # Set up device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Create subject-specific save directory
    subject_dir = save_dir / f"subject_{subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = ModelFactory.create(config.model_name, config)
    model = model.to(device)
    
    # Load base model weights
    checkpoint = torch.load(base_model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Update config for subject-specific training
    config.subject_id = subject_id
    config.learning_rate = config.finetune_learning_rate
    config.num_epochs = config.finetune_epochs
    
    # Get subject-specific dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Verify subject filtering
    train_subjects = pd.unique(train_loader.dataset.df['sub'])
    val_subjects = pd.unique(val_loader.dataset.df['sub'])
    test_subjects = pd.unique(test_loader.dataset.df['sub'])
    
    print(f"\nSubjects in training set: {train_subjects}")
    print(f"Subjects in validation set: {val_subjects}")
    print(f"Subjects in test set: {test_subjects}")
    
    if len(train_subjects) != 1 or train_subjects[0] != subject_id:
        raise ValueError(f"Training set contains incorrect subjects: {train_subjects}")
    if len(val_subjects) != 1 or val_subjects[0] != subject_id:
        raise ValueError(f"Validation set contains incorrect subjects: {val_subjects}")
    if len(test_subjects) != 1 or test_subjects[0] != subject_id:
        raise ValueError(f"Test set contains incorrect subjects: {test_subjects}")
    
    # Check if best model exists
    best_model_path = subject_dir / "best_model.pt"
    
    # Look for existing test results
    test_results_dirs = list(subject_dir.glob("best_model_test_results_*/test_results.csv"))
    if test_results_dirs and best_model_path.exists():
        # Extract timestamps and find the most recent one
        def get_timestamp(path):
            # Extract timestamp from directory name (format: YYYYMMDD_HHMMSS)
            timestamp_str = path.parent.name.split('_')[-2] + '_' + path.parent.name.split('_')[-1]
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        
        # Get the most recent test results file
        latest_results = max(test_results_dirs, key=get_timestamp)
        print(f"\nFound {len(test_results_dirs)} test results for subject {subject_id}")
        print(f"Using most recent results from: {latest_results.parent.name}")
        print("Loading existing results instead of re-testing.")
        return pd.read_csv(latest_results)
    
    if best_model_path.exists():
        print(f"\nBest model exists for subject {subject_id}. Skipping training and proceeding to testing.")
        return test_model(str(best_model_path), config)
    
    # Initialize trainer for fine-tuning
    trainer = EnvelopeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=LossFactory.create(config.loss_function),
        optimizer=torch.optim.Adam(model.parameters(), lr=config.finetune_learning_rate),
        device=device,
        config=config.to_dict(),
        experiment_name=f"fine_tune_subject_{subject_id}",
        checkpoint_dir=str(subject_dir),
        subject_id=subject_id,
        wandb_project=config.wandb_project,
        wandb_tags=[
            f"model:{config.model_name}",
            f"loss:{config.loss_function}",
            f"subject:{subject_id}",
            "phase:fine_tuning"
        ]
    )
    
    # Fine-tune the model
    trainer.train(num_epochs=config.finetune_epochs)
    
    # Test the fine-tuned model
    best_model_path = subject_dir / "best_model.pt"
    test_results = test_model(str(best_model_path), config)
    
    return test_results

def create_comparison_plots(base_results: pd.DataFrame, fine_tuned_results: pd.DataFrame, save_dir: Path):
    """Create plots comparing base model and fine-tuned model performance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Subject': base_results['subject_id'].tolist() + fine_tuned_results['subject_id'].tolist(),
        'Correlation': base_results['correlation'].tolist() + fine_tuned_results['correlation'].tolist(),
        'Model': ['Base']*len(base_results) + ['Fine-tuned']*len(fine_tuned_results)
    })
    
    # Create comparison violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_data, x='Model', y='Correlation')
    plt.title('Distribution of Correlations: Base vs Fine-tuned')
    plt.tight_layout()
    violin_path = save_dir / f"model_comparison_violin_{timestamp}.png"
    plt.savefig(violin_path)
    plt.close()
    
    # Create subject-wise comparison plot
    plt.figure(figsize=(15, 8))
    subjects = sorted(base_results['subject_id'].unique())
    base_means = [base_results[base_results['subject_id'] == sub]['correlation'].mean() for sub in subjects]
    fine_tuned_means = [fine_tuned_results[fine_tuned_results['subject_id'] == sub]['correlation'].mean() for sub in subjects]
    
    x = np.arange(len(subjects))
    width = 0.35
    
    plt.bar(x - width/2, base_means, width, label='Base Model')
    plt.bar(x + width/2, fine_tuned_means, width, label='Fine-tuned')
    
    plt.xlabel('Subject')
    plt.ylabel('Mean Correlation')
    plt.title('Subject-wise Performance Comparison')
    plt.xticks(x, subjects, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    comparison_path = save_dir / f"subject_comparison_bars_{timestamp}.png"
    plt.savefig(comparison_path)
    plt.close()
    
    return violin_path, comparison_path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for each subject")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to base model checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline_config.py",
                      help="Path to config file")
    parser.add_argument("--subjects", type=str, default=None,
                      help="Comma-separated list of subject IDs to fine-tune (e.g., 'sub-001,sub-002'). If not provided, will fine-tune all subjects.")
    parser.add_argument("--start-subject", type=int, default=None,
                      help="Start subject number (1-85). If provided with --end-subject, will fine-tune this range.")
    parser.add_argument("--end-subject", type=int, default=None,
                      help="End subject number (1-85). If provided with --start-subject, will fine-tune this range.")
    args = parser.parse_args()
    
    # Load config
    config = BaselineConfig()
    
    # Set up directories
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent  # Go up two levels from the checkpoint
    fine_tune_dir = base_dir / "fine_tuned_models"
    fine_tune_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which subjects to fine-tune
    if args.subjects:
        # Use explicitly provided subject IDs
        all_subjects = args.subjects.split(',')
    elif args.start_subject is not None and args.end_subject is not None:
        # Generate range of subjects
        all_subjects = [f"sub-{i:03d}" for i in range(args.start_subject, args.end_subject + 1)]
    else:
        # Default: all 85 subjects
        all_subjects = [f"sub-{i:03d}" for i in range(1, 86)]
    
    print(f"\nPreparing to fine-tune {len(all_subjects)} subjects:")
    print(f"First subject: {all_subjects[0]}")
    print(f"Last subject: {all_subjects[-1]}")
    
    # First, test the base model for all subjects
    print("\nTesting base model performance...")
    config.subject_id = None  # Ensure we get all subjects for testing
    base_results = test_model(str(checkpoint_path), config)
    
    # Add subject IDs to base results from the 'sub' column
    if 'sub' in base_results.columns:
        base_results['subject_id'] = base_results['sub']
    
    base_results_path = fine_tune_dir / "base_model_results.csv"
    base_results.to_csv(base_results_path, index=False)
    
    # Store results for all subjects
    fine_tuned_results = []
    
    # Fine-tune for each subject
    for subject_id in all_subjects:
        print(f"\n{'='*80}")
        print(f"Fine-tuning for subject: {subject_id}")
        print(f"{'='*80}")
        
        try:
            results = fine_tune_subject(
                base_model_checkpoint=str(checkpoint_path),
                subject_id=subject_id,
                config=config,
                save_dir=fine_tune_dir
            )
            
            # Add subject ID to results if it's a DataFrame
            if isinstance(results, pd.DataFrame):
                results['subject_id'] = subject_id
            
            fine_tuned_results.append(results)
            
        except Exception as e:
            print(f"Error fine-tuning subject {subject_id}: {str(e)}")
            continue
    
    # Save and analyze results
    if fine_tuned_results:
        # Combine all results
        all_results = pd.concat(fine_tuned_results)
        
        # Add subject ID from 'sub' column if needed
        if 'sub' in all_results.columns and 'subject_id' not in all_results.columns:
            all_results['subject_id'] = all_results['sub']
        
        summary_path = fine_tune_dir / "fine_tuning_summary.csv"
        all_results.to_csv(summary_path, index=False)
        
        # Create comparison plots
        violin_path, comparison_path = create_comparison_plots(
            base_results,
            all_results,
            fine_tune_dir
        )
        
        # Calculate summary statistics
        subject_means = all_results.groupby('subject_id')['correlation'].mean()
        mean_of_means = subject_means.mean()
        std_of_means = subject_means.std()
        sem = std_of_means / np.sqrt(len(subject_means))
        
        # Save detailed summary
        summary_txt_path = fine_tune_dir / "fine_tuning_analysis.txt"
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

if __name__ == "__main__":
    main() 