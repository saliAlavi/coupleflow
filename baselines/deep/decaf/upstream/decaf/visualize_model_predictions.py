import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.eeg_dataloader import create_dataloader, StandardizeEEG, StandardizeEnvelope
from src.models.model_factory import ModelFactory
import seaborn as sns
from src.models import registry  # This will register all models
import pandas as pd
from src.train.weighted_losses import WeightedPearsonLoss
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Optional, Tuple
import re
import itertools
import pickle

def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    """Load a model from checkpoint."""
    try:
        # Get config class and create instance
        ConfigClass = ModelFactory.get_config_class(model_type)
        config = ConfigClass()
        config.model_name = model_type
        
        # Create model using factory
        model = ModelFactory.create(model_type, config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model {model_type}: {str(e)}")
        raise e

def normalize_signal(signal):
    """Normalize signal to zero mean and unit variance."""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)

def generate_constant_masks(batch_size: int, seq_len: int, context_seconds: float, sampling_rate: int = 64) -> torch.Tensor:
    """
    Generate deterministic masks based on context window.
    
    Args:
        batch_size: Number of samples in the batch
        seq_len: Length of each sequence
        context_seconds: Number of seconds to use as context
        sampling_rate: Sampling rate in Hz (default: 64)
        
    Returns:
        mask_matrix: Boolean mask of shape (batch_size, seq_len)
            True indicates positions to mask
    """
    # Initialize mask matrix
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Calculate context samples
    context_samples = int(context_seconds * sampling_rate)
    
    # Mask everything after the context window
    mask_matrix[:, context_samples:] = True
    
    return mask_matrix

def get_model_info(model_name: str) -> tuple:
    """Extract base model name and loss function from model name."""
    # Extract base model name
    base_model = model_name.split('_')[0]
    
    # Determine loss function
    if 'mse' in model_name.lower():
        loss_type = 'MSE'
    elif 'pearson_l1' in model_name.lower():
        loss_type = 'Pearson-L1'
    elif 'pearson' in model_name.lower():
        loss_type = 'Pearson'
    else:
        loss_type = 'Unknown'
    
    return base_model, loss_type

def visualize_predictions(models: dict, test_batches: dict, save_path: str = None, models_info: List[Dict] = None):
    """
    Visualize predictions from multiple models against ground truth.
    Shows 10 examples with predictions from all models, grouped by model type and loss function.
    
    Args:
        models: Dictionary of model_name: model pairs
        test_batches: Dictionary of model_name: test_batch pairs, where each test_batch is a tuple of (metadata, eegs, envelopes, envelope_context)
        save_path: Path to save the visualization
        models_info: List of dictionaries containing model information
    """
    device = next(iter(models.values())).parameters().__next__().device
    
    # Save envelope context for each model
    for name, test_batch in test_batches.items():
        metadata, eegs, envelopes, eeg_context, envelope_context = test_batch
        if envelope_context is not None:
            # Extract context length from model name if possible
            context_length = "default"
            if models_info:
                for info in models_info:
                    if info['name'] == name and 'context_seconds' in info:
                        context_length = f"{info['context_seconds']}"
                        break
    
    # Get predictions from all models
    predictions = {}
    ground_truths = {}
    model_groups = {}
    
    with torch.no_grad():
        for name, model in models.items():
            # Get the model-specific test batch
            if name not in test_batches:
                print(f"Warning: No test batch found for model {name}. Skipping.")
                continue
                
            test_batch = test_batches[name]
            metadata, eegs, envelopes, eeg_context, envelope_context = test_batch

            print(f"Processing model {name} with context {envelope_context.shape}")
            print(f"EEG shape: {eegs.shape}")
            print(f"Envelope shape: {envelopes.shape}")
            print(f"Envelope context shape: {envelope_context.shape}")
            
            try:
                outputs = model(eegs.to(device))
            except Exception as ex:
                try:
                    outputs = model(eegs.to(device), envelope_context.to(device))
                    print(model.envelope_branch.prediction_seconds, model.envelope_branch.context_seconds)
                except Exception as ex2:
                    print(f"Error processing model {name}: {ex2}")
                    continue
            # Get predictions from outputs dictionary
            pred = outputs['envelope']
            predictions[name] = pred.cpu().numpy()
            ground_truths[name] = envelopes.numpy()
    
            # Group models by base type and loss function.
            base_model, loss_type = get_model_info(name)
            if base_model not in model_groups:
                model_groups[base_model] = {}
            if loss_type not in model_groups[base_model]:
                model_groups[base_model][loss_type] = []
            model_groups[base_model][loss_type].append(name)
    
    # --- New cropping: Select only the last portion using the constant CROP_END ---
    for key in predictions:
        predictions[key] = predictions[key][:, :CROP_END, :]
        ground_truths[key] = ground_truths[key][:, :CROP_END, :]
    # ---------------------------------------------------------------------------

    print(predictions.keys())
    
    # Find valid examples (where all models have valid correlations)
    valid_examples = []
    for i in range(predictions[next(iter(predictions.keys()))].shape[0]):
        example_valid = True
        for name, pred in predictions.items():
            gt = ground_truths[name]
            try:
                normalized_gt = normalize_signal(gt[i, :, 0])
                normalized_pred = normalize_signal(pred[i, :, 0])
                correlation = np.corrcoef(normalized_gt, normalized_pred)[0, 1]
                if not np.isfinite(correlation) or np.isnan(correlation):
                    example_valid = False
                    break
            except Exception:
                example_valid = False
                break
        if example_valid:
            valid_examples.append(i)
    
    # Use only valid examples, up to 10
    num_examples = min(10, len(valid_examples))
    if num_examples == 0:
        print("No valid examples to visualize. Skipping visualization.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    # Define colors for different models
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    # Create a legend mapping
    legend_handles = []
    legend_labels = []
    
    # Plot each example
    for plot_idx, example_idx in enumerate(valid_examples[:num_examples]):
        ax = axes[plot_idx]
        
        # Get ground truth for this example (use first model's ground truth)
        first_gt = next(iter(ground_truths.values()))
        time = np.arange(first_gt.shape[1]) / 64.0  # Assuming 64 Hz sampling rate
        normalized_gt = normalize_signal(first_gt[example_idx, :, 0])
        
        # Plot ground truth
        ax.plot(time, normalized_gt, 'k-', label='Ground Truth', linewidth=2)
        
        # Calculate y-axis limits before adding text
        y_min, y_max = normalized_gt.min(), normalized_gt.max()
        
        # Plot each model's prediction
        for j, (name, pred) in enumerate(predictions.items()):
            # Get corresponding ground truth
            gt = ground_truths[name]
            
            # Plot prediction
            base_model, loss_type = get_model_info(name)
            color = colors[color_idx % len(colors)]
            
            # Add context information to the label if available
            context_info = ""
            if models_info:
                for info in models_info:
                    if info['name'] == name and 'context_seconds' in info:
                        context_info = f" (Context: {info['context_seconds']}s)"
                        break
            
            # Calculate correlation for this example
            normalized_pred = normalize_signal(pred[example_idx, :, 0])
            correlation = np.corrcoef(normalized_gt, normalized_pred)[0, 1]
            
            # Update y-axis limits
            y_min = min(y_min, normalized_pred.min())
            y_max = max(y_max, normalized_pred.max())
            
            # Create label without correlation (it will be added to the plot separately)
            label = f"{base_model} ({loss_type}){context_info}"
            
            # Plot prediction
            ax.plot(time, normalized_pred, '-', color=color, 
                    label=label, linewidth=1.5, alpha=0.8)
            
            # Add correlation text on the right side of the plot
            # Position text vertically based on the number of models
            text_y = y_max - (j + 1) * (y_max - y_min) / (len(predictions) + 1)
            ax.text(time[-1] + 0.1, text_y, 
                   f"{base_model}: r = {correlation:.3f}", 
                   color=color, va='center', fontsize=10)
            
            # Add to legend mapping if first example
            if plot_idx == 0:
                legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.5))
                legend_labels.append(label)
            
            color_idx += 1
        
        # Adjust plot limits to accommodate correlation text
        ax.set_xlim(time[0], time[-1] + 1.0)  # Add space for correlation text
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        
        # Add title and labels
        metadata_example = next(iter(test_batches.values()))[0]  # Get metadata from first test batch
        if 'sub' in metadata_example and 'audio_file' in metadata_example:
            ax.set_title(f"Example {plot_idx+1}: Subject {metadata_example['sub'][example_idx]}, Audio {metadata_example['audio_file'][example_idx]}")
        else:
            ax.set_title(f"Example {plot_idx+1}")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    
    # Add a single legend for the entire figure
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=min(5, len(legend_labels)), frameon=True)
    
    # Add a title to the figure indicating model-specific test batches are used
    fig.suptitle("Model Predictions with Model-Specific Test Batches", fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.85)  # Make room for the legend and correlation text
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def get_subject_means(predictions: dict, ground_truth: np.ndarray, metadata: list) -> Dict[str, pd.Series]:
    """Extract mean performance per subject for each model."""
    subject_means = {}
    for name, pred in predictions.items():
        # Calculate correlations for each sample
        correlations = []
        subjects = []
        for i in range(len(pred)):
            corr = np.corrcoef(ground_truth[i, :, 0], pred[i, :, 0])[0, 1]
            correlations.append(corr)
            subjects.append(metadata['sub'][i])
        
        # Create DataFrame and compute mean per subject
        df = pd.DataFrame({'subject': subjects, 'correlation': correlations})
        subject_means[name] = df.groupby('subject')['correlation'].mean()
    
    return subject_means

def save_statistical_results(comparisons: List[tuple], p_values: List[float], p_corrected: List[float], 
                          reject: List[bool], effect_sizes: List[float], subject_means: Dict[str, pd.Series],
                          save_path: str = None):
    """Save statistical test results to a file."""
    if save_path is None:
        save_path = f"{SAVE_DIR}/statistical_test_results.txt"
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Statistical Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        
        # First, write individual model statistics
        f.write("Individual Model Statistics:\n")
        f.write("-" * 80 + "\n\n")
        
        for model_name, means in subject_means.items():
            # Extract base model name without context info
            base_model_name = model_name.split(" (Context:")[0]
            
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall correlation: {means.mean():.4f} (±{means.std():.4f})\n")
            f.write(f"Median correlation: {means.median():.4f}\n")
            f.write(f"Min correlation: {means.min():.4f}\n")
            f.write(f"Max correlation: {means.max():.4f}\n")
            f.write(f"Number of subjects: {len(means)}\n")
            f.write(f"Best subject: {means.idxmax()} (r={means.max():.4f})\n")
            f.write(f"Worst subject: {means.idxmin()} (r={means.min():.4f})\n\n")
        
        # Then, write pairwise comparison results
        f.write("\nPairwise Comparisons (Wilcoxon signed-rank test with Bonferroni correction):\n")
        f.write("-" * 80 + "\n\n")
        
        for idx, ((model1, model2), p_orig, p_corr, rej, effect) in enumerate(zip(comparisons, p_values, p_corrected, reject, effect_sizes)):
            # Extract base model names without context info for mean calculation
            base_model1 = model1.split(" (Context:")[0]
            base_model2 = model2.split(" (Context:")[0]
            
            # Make sure the base model names exist in subject_means
            if base_model1 not in subject_means:
                base_model1 = model1
            if base_model2 not in subject_means:
                base_model2 = model2
            
            mean1 = subject_means[base_model1].mean()
            mean2 = subject_means[base_model2].mean()
            
            f.write(f"Comparison {idx + 1}: {model1} vs {model2}\n")
            f.write(f"Mean correlation: {mean1:.4f} vs {mean2:.4f}\n")
            f.write(f"Original p-value: {p_orig:.4e}\n")
            f.write(f"Corrected p-value: {p_corr:.4e}\n")
            f.write(f"Effect size (r): {effect:.4f}\n")
            f.write(f"Significant: {'Yes' if rej else 'No'}\n")
            f.write(f"Better model: {model1 if mean1 > mean2 else model2}\n\n")

def compare_models_statistically(predictions: dict, ground_truth: np.ndarray, metadata: list, models_info: List[Dict] = None):
    """Compare models statistically using Wilcoxon signed-rank test."""
    # Calculate mean correlation per subject
    subject_means = get_subject_means(predictions, ground_truth, metadata)
    
    # Get all model pairs
    models = list(subject_means.keys())
    pairs = list(itertools.combinations(models, 2))
    
    # Perform Wilcoxon signed-rank test for each pair
    p_values = []
    comparisons = []
    effect_sizes = []
    
    for model_a, model_b in pairs:
        # Get context information if available
        context_a = ""
        context_b = ""
        if models_info:
            model_a_info = next((info for info in models_info if info['name'] == model_a), None)
            model_b_info = next((info for info in models_info if info['name'] == model_b), None)
            if model_a_info and 'context_seconds' in model_a_info:
                context_a = f" (Context: {model_a_info['context_seconds']}s)"
            if model_b_info and 'context_seconds' in model_b_info:
                context_b = f" (Context: {model_b_info['context_seconds']}s)"
                
        # Perform the test
        a_values = subject_means[model_a]
        b_values = subject_means[model_b]
        
        # Apply Wilcoxon signed-rank test
        stat, p_value = wilcoxon(a_values, b_values, alternative="two-sided")
        
        # Calculate effect size (r = Z / sqrt(N))
        effect_size = stat / np.sqrt(len(a_values))
        
        # Store results
        p_values.append(p_value)
        comparisons.append((f"{model_a}{context_a}", f"{model_b}{context_b}"))
        effect_sizes.append(effect_size)
    
    # Apply Bonferroni correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    
    # Save results to file
    save_path = f"{SAVE_DIR}/statistical_test_results.txt"
    save_statistical_results(comparisons, p_values, p_corrected, reject, effect_sizes, subject_means, save_path)
    
    print(f"\nSaved statistical test results to {save_path}")

def print_model_statistics(predictions: dict, ground_truth: np.ndarray, metadata: list, models_info: List[Dict]):
    """Print statistics for each model."""
    # Calculate mean correlation per subject
    subject_means = get_subject_means(predictions, ground_truth, metadata)
    
    for model_name, means in subject_means.items():
        # Find the corresponding model info
        model_info = next((info for info in models_info if info['name'] == model_name), None)
        context_str = ""
        if model_info and 'context_seconds' in model_info:
            context_str = f" (Context: {model_info['context_seconds']}s)"
            
        print(f"\nModel: {model_name}{context_str}")
        print("-" * 40)
        print(f"Overall correlation: {means.mean():.4f} (±{means.std():.4f})")
        print(f"Median correlation: {means.median():.4f}")
        print(f"Min correlation: {means.min():.4f}")
        print(f"Max correlation: {means.max():.4f}")
        print(f"Number of subjects: {len(means)}")
        print(f"Best subject: {means.idxmax()} (r={means.max():.4f})")
        print(f"Worst subject: {means.idxmin()} (r={means.min():.4f})")

def get_simplified_model_label(model_name: str, model_info: Dict) -> str:
    """Create a simplified model label showing only model type and loss function."""
    # Find the corresponding model info
    info = next((info for info in model_info if info['name'] == model_name), None)
    if info:
        # Get the base model name (without the version number)
        base_name = info['dir_type'].replace('2', '').replace('3', '').replace('4', '')
        return f"{base_name}\n({info['loss']})"
    return model_name

def create_violin_plot(stats_df: pd.DataFrame, save_path: str, models_info: List[Dict]):
    """Create violin plot with statistical significance between models."""
    # Select only numeric columns (performance metrics)
    numeric_data = stats_df.iloc[:, 4:]
    
    # Create three versions of the plot: 
    # 1. Grouped by subject
    # 2. Non-grouped (all samples)
    # 3. Grouped by audio segment (audio_file, start, end)
    
    # 1. Grouped by subject (original plot)
    # Group by 'subject' and calculate the mean for each metric
    grouped_data = stats_df.groupby('subject')[numeric_data.columns].mean()
    
    # Group models by architecture and context length
    architecture_groups = {}
    for col in grouped_data.columns:
        # Find corresponding model info
        model_info = next((info for info in models_info if info['name'] == col), None)
        if model_info:
            # Get base architecture type
            dir_type = model_info['dir_type']
            arch_type = dir_type.split('_')[0]  # Get first part before underscore
            
            # Remove version numbers and get base architecture
            clean_arch = ''.join([c for c in arch_type if not c.isdigit()])
            
            # Include context length in the architecture key if available
            context_key = ""
            if 'context_seconds' in model_info:
                context_key = f"_{model_info['context_seconds']}s"
            
            # Create a unique key combining architecture and context length
            combined_key = f"{clean_arch}{context_key}"
            
            if combined_key not in architecture_groups:
                architecture_groups[combined_key] = []
            architecture_groups[combined_key].append({
                'name': col,
                'mean_perf': grouped_data[col].mean(),
                'full_info': model_info
            })
    
    # Select best model from each architecture+context group
    best_models = []
    for arch_type, models in architecture_groups.items():
        # Sort models by mean performance
        sorted_models = sorted(models, key=lambda x: x['mean_perf'], reverse=True)
        # Select the best model
        print(sorted_models[0])
        best_models.append(sorted_models[0])
    
    # Sort best models by overall performance
    best_models.sort(key=lambda x: x['mean_perf'])
    
    # Create filtered dataframe with only the best models
    best_model_names = [model['name'] for model in best_models]
    filtered_data = grouped_data[best_model_names]
    
    # Create full labels with architecture and loss type
    full_labels = []
    for model in best_models:
        # Get a clean architecture name without version numbers
        dir_type = model['full_info']['dir_type']
        arch_type = dir_type.split('_')[0]  # Get first part before underscore
        
        # Remove version numbers and capitalize for display
        clean_arch = ''.join([c for c in arch_type if not c.isdigit()])
        clean_arch = clean_arch.capitalize()
        
        # For baseline decoder, make it more readable
        if 'baseline' in clean_arch.lower():
            clean_arch = 'Baseline'
            
        loss_type = model['full_info']['loss']
        
        # Always include context information in the label
        context_info = ""
        if 'context_seconds' in model['full_info']:
            context_info = f"\nContext: {model['full_info']['context_seconds']}s"
        else:
            context_info = f"\nContext: default"
            
        full_labels.append(f"{clean_arch}\n({loss_type}){context_info}")
    
    # Perform Wilcoxon signed-rank test between consecutive models
    p_values = []
    for i in range(len(best_model_names) - 1):
        model_a = filtered_data[best_model_names[i]]
        model_b = filtered_data[best_model_names[i + 1]]
        # Apply Wilcoxon signed-rank test
        stat, p_value = wilcoxon(model_a, model_b, alternative="two-sided")
        p_values.append(p_value)
    
    # Create the grouped violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=filtered_data, inner="quartile")
    
    # Add p-values for significance
    for i, p_value in enumerate(p_values):
        x_position = i + 0.5  # Position between the two models
        y_position = max(filtered_data.iloc[:, i].max(),
                        filtered_data.iloc[:, i + 1].max()) + 0.02
        plt.text(x_position, y_position, f"p={p_value:.3g}", 
                ha='center', fontsize=10, color='blue')
    
    # Customize the plot
    plt.xticks(range(len(full_labels)), full_labels, rotation=45, ha='right')
    plt.title("Best Model Performance by Architecture and Context Length (Grouped by Subject)", fontsize=16)
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Correlation Coefficient (r)", fontsize=14)
    
    # Add mean performance values
    for i, model in enumerate(best_models):
        mean_perf = model['mean_perf']
        plt.text(i, filtered_data.min().min() - 0.05, 
                f"μ={mean_perf:.3f}", 
                ha='center', va='top', 
                fontsize=10, color='darkred')
    
    plt.tight_layout()
    
    # Save the grouped plot
    grouped_save_path = save_path.replace('.png', '_grouped.png')
    plt.savefig(grouped_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved grouped violin plot to {grouped_save_path}")
    
    # 2. Non-grouped violin plot (using all individual correlations)
    # Prepare data for non-grouped plot
    non_grouped_data = []
    for model_name in best_model_names:
        correlations = stats_df[model_name].values
        for corr in correlations:
            non_grouped_data.append({
                'Model': model_name,
                'Correlation': corr
            })
    
    non_grouped_df = pd.DataFrame(non_grouped_data)
    
    # Create the non-grouped violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=non_grouped_df, x='Model', y='Correlation', inner="quartile")
    
    # Perform statistical tests between consecutive models
    p_values_non_grouped = []
    for i in range(len(best_model_names) - 1):
        model_a_data = non_grouped_df[non_grouped_df['Model'] == best_model_names[i]]['Correlation']
        model_b_data = non_grouped_df[non_grouped_df['Model'] == best_model_names[i + 1]]['Correlation']
        # Apply Mann-Whitney U test for independent samples
        _, p_value = stats.mannwhitneyu(model_a_data, model_b_data, alternative="two-sided")
        p_values_non_grouped.append(p_value)
    
    # Add p-values for significance
    for i, p_value in enumerate(p_values_non_grouped):
        x_position = i + 0.5
        y_position = max(non_grouped_df[non_grouped_df['Model'] == best_model_names[i]]['Correlation'].max(),
                        non_grouped_df[non_grouped_df['Model'] == best_model_names[i + 1]]['Correlation'].max()) + 0.02
        plt.text(x_position, y_position, f"p={p_value:.3g}", 
                ha='center', fontsize=10, color='blue')
    
    # Customize the plot
    plt.xticks(range(len(full_labels)), full_labels, rotation=45, ha='right')
    plt.title("Best Model Performance by Architecture and Context Length (All Samples)", fontsize=16)
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Correlation Coefficient (r)", fontsize=14)
    
    # Add mean performance values
    for i, model_name in enumerate(best_model_names):
        mean_perf = non_grouped_df[non_grouped_df['Model'] == model_name]['Correlation'].mean()
        plt.text(i, non_grouped_df['Correlation'].min() - 0.05, 
                f"μ={mean_perf:.3f}", 
                ha='center', va='top', 
                fontsize=10, color='darkred')
    
    plt.tight_layout()
    
    # Save the non-grouped plot
    non_grouped_save_path = save_path.replace('.png', '_non_grouped.png')
    plt.savefig(non_grouped_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved non-grouped violin plot to {non_grouped_save_path}")
    
    # 3. NEW: Grouped by audio segment (audio_file, start, end)
    # Create a unique identifier for each audio segment
    stats_df['audio_segment'] = stats_df['audio_file'] + '_' + stats_df['start'].astype(str) + '_' + stats_df['end'].astype(str)
    
    # Group by audio segment and calculate the mean for each metric
    audio_grouped_data = stats_df.groupby('audio_segment')[numeric_data.columns].mean()
    
    # Filter to include only the best models
    audio_filtered_data = audio_grouped_data[best_model_names]
    
    # Perform Wilcoxon signed-rank test between consecutive models
    p_values_audio = []
    for i in range(len(best_model_names) - 1):
        model_a = audio_filtered_data[best_model_names[i]]
        model_b = audio_filtered_data[best_model_names[i + 1]]
        # Apply Wilcoxon signed-rank test
        stat, p_value = wilcoxon(model_a, model_b, alternative="two-sided")
        p_values_audio.append(p_value)
    
    # Create the audio-grouped violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=audio_filtered_data, inner="quartile")
    
    # Add p-values for significance
    for i, p_value in enumerate(p_values_audio):
        x_position = i + 0.5  # Position between the two models
        y_position = max(audio_filtered_data.iloc[:, i].max(),
                        audio_filtered_data.iloc[:, i + 1].max()) + 0.02
        plt.text(x_position, y_position, f"p={p_value:.3g}", 
                ha='center', fontsize=10, color='blue')
    
    # Customize the plot
    plt.xticks(range(len(full_labels)), full_labels, rotation=45, ha='right')
    plt.title("Best Model Performance by Architecture and Context Length (Grouped by Audio Segment)", fontsize=16)
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Correlation Coefficient (r)", fontsize=14)
    
    # Add mean performance values
    for i, model_name in enumerate(best_model_names):
        mean_perf = audio_filtered_data[model_name].mean()
        plt.text(i, audio_filtered_data.min().min() - 0.05, 
                f"μ={mean_perf:.3f}", 
                ha='center', va='top', 
                fontsize=10, color='darkred')
    
    plt.tight_layout()
    
    # Save the audio-grouped plot
    audio_grouped_save_path = save_path.replace('.png', '_audio_grouped.png')
    plt.savefig(audio_grouped_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved audio-grouped violin plot to {audio_grouped_save_path}")
    
    # 4. NEW: Grouped by subject with balanced audio file representation
    # First group by subject and audio_file to get mean performance for each subject-audio_file combination
    subject_audio_grouped = stats_df.groupby(['subject', 'audio_file'])[numeric_data.columns].mean()
    
    # Then group by subject to get the mean across all audio files for each subject
    balanced_subject_grouped = subject_audio_grouped.groupby('subject').mean()
    
    # Filter to include only the best models
    balanced_filtered_data = balanced_subject_grouped[best_model_names]
    
    # Perform Wilcoxon signed-rank test between consecutive models
    p_values_balanced = []
    for i in range(len(best_model_names) - 1):
        model_a = balanced_filtered_data[best_model_names[i]]
        model_b = balanced_filtered_data[best_model_names[i + 1]]
        # Apply Wilcoxon signed-rank test
        stat, p_value = wilcoxon(model_a, model_b, alternative="two-sided")
        p_values_balanced.append(p_value)
    
    # Create the balanced subject-grouped violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=balanced_filtered_data, inner="quartile")
    
    # Add p-values for significance
    for i, p_value in enumerate(p_values_balanced):
        x_position = i + 0.5  # Position between the two models
        y_position = max(balanced_filtered_data.iloc[:, i].max(),
                        balanced_filtered_data.iloc[:, i + 1].max()) + 0.02
        plt.text(x_position, y_position, f"p={p_value:.3g}", 
                ha='center', fontsize=10, color='blue')
    
    # Customize the plot
    plt.xticks(range(len(full_labels)), full_labels, rotation=45, ha='right')
    plt.title("Best Model Performance (Balanced by Audio File per Subject)", fontsize=16)
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Correlation Coefficient (r)", fontsize=14)
    
    # Add mean performance values
    for i, model_name in enumerate(best_model_names):
        mean_perf = balanced_filtered_data[model_name].mean()
        plt.text(i, balanced_filtered_data.min().min() - 0.05, 
                f"μ={mean_perf:.3f}", 
                ha='center', va='top', 
                fontsize=10, color='darkred')
    
    plt.tight_layout()
    
    # Save the balanced subject-grouped plot
    balanced_save_path = save_path.replace('.png', '_balanced_subject.png')
    plt.savefig(balanced_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved balanced subject-grouped violin plot to {balanced_save_path}")
    
    # Print summary of best models
    print("\nBest Models by Architecture:")
    print("=" * 80)
    for model in best_models:
        print(f"\nArchitecture: {model['full_info']['dir_type'].capitalize()}")
        print(f"Model: {model['name']}")
        print(f"Loss: {model['full_info']['loss']}")
        print(f"Mean Performance: {model['mean_perf']:.4f}")

def discover_models(checkpoint_dir: str, model_mappings: Dict[str, str]) -> List[Dict]:
    """
    Discover all model checkpoints in the given directory.
    
    Args:
        checkpoint_dir: Path to the directory containing model checkpoints
        model_mappings: Dictionary mapping directory names to registry model names
        
    Returns:
        List of dictionaries containing model info (name, path, type)
    """
    models_info = []
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    for model_dir in model_dirs:
        if model_dir == "model_comparison_analysis":  # Skip analysis directory
            continue
            
        # Process subject-independent results
        subject_independent_dir = os.path.join(checkpoint_dir, model_dir, "subject_independent")
        if not os.path.exists(subject_independent_dir):
            continue
            
        # Get all run directories
        run_dirs = [d for d in os.listdir(subject_independent_dir) 
                   if os.path.isdir(os.path.join(subject_independent_dir, d))]
        
        for run_dir in run_dirs:
            checkpoint_path = os.path.join(subject_independent_dir, run_dir, "best_model.pt")
            if os.path.exists(checkpoint_path):
                # Extract model type from directory name
                model_dir_type = model_dir.lower()  # Use the full directory name
                
                # Map to registry model name using model_mappings. If not present, skip.
                # First try to match the full directory name, then try the first part
                registry_key = model_dir_type
                if registry_key not in model_mappings:
                    registry_key = model_dir_type.split('_')[0]
                    if registry_key not in model_mappings:
                        print(f"Warning: No mapping found for model type {model_dir_type}")
                        continue
                    
                model_type = model_mappings[registry_key]
                
                # Create display name
                if 'mse' in model_dir.lower():
                    loss_type = 'MSE'
                elif 'pearson_l1' in model_dir.lower():
                    loss_type = 'Pearson-L1'
                elif 'pearson' in model_dir.lower():
                    loss_type = 'Pearson'
                else:
                    loss_type = 'Unknown'
                
                display_name = f"{model_dir}_{run_dir}"
                
                models_info.append({
                    'name': display_name,
                    'path': checkpoint_path,
                    'type': model_type,  # Use mapped model type
                    'dir_type': model_dir_type,  # Keep original directory type for grouping
                    'loss': loss_type
                })
    
    return models_info

def load_all_models(models_info: List[Dict], device: torch.device) -> Dict:
    """
    Load all discovered models.
    
    Args:
        models_info: List of model information dictionaries
        device: Device to load models to
        
    Returns:
        Dictionary of loaded models
    """
    loaded_models = {}
    
    for model_info in models_info:
        try:
            print(f"\nLoading model: {model_info['name']}")
            model = load_model(model_info['path'], model_info['type'], device)
            loaded_models[model_info['name']] = model
            print(f"Successfully loaded {model_info['name']}")
        except Exception as e:
            print(f"Error loading {model_info['name']}: {str(e)}")
            continue
    
    return loaded_models

def save_model_data(models: dict, test_batches: dict, save_path: str, models_info: List[Dict] = None):
    """
    Save inputs, ground truths, and outputs from all models for later analysis.
    
    Args:
        models: Dictionary of model_name: model pairs
        test_batches: Dictionary of model_name: test_batch pairs, where each test_batch is a tuple of (metadata, eegs, envelopes, eeg_context, envelope_context)
        save_path: Path to save the data
        models_info: List of dictionaries containing model information
    """
    device = next(iter(models.values())).parameters().__next__().device
    
    # Dictionary to store all data
    all_data = {
        'metadata': [],
        'inputs': {},
        'ground_truths': {},
        'envelope_contexts': {},  # Added envelope_contexts
        'outputs': {}
    }
    
    with torch.no_grad():
        for name, model in models.items():
            # Get the model-specific test batch
            if name not in test_batches:
                print(f"Warning: No test batch found for model {name}. Skipping.")
                continue
                
            test_batch = test_batches[name]
            metadata, eegs, envelopes, eeg_context, envelope_context = test_batch
            
            # Store metadata for the first model only
            if not all_data['metadata']:
                for i in range(len(metadata['sub'])):
                    all_data['metadata'].append({
                        'subject': metadata['sub'][i],
                        'audio_file': metadata['audio_file'][i],
                        'paths_eeg': metadata['paths_eeg'][i],
                        'paths_envelope': metadata['paths_envelope'][i],
                        'start': metadata['start'][i].item(),
                        'end': metadata['end'][i].item()
                    })
            
            # Store inputs
            all_data['inputs'][name] = eegs.cpu().numpy()
            
            # Store ground truths
            all_data['ground_truths'][name] = envelopes.cpu().numpy()
            
            # Store envelope contexts if available
            if envelope_context is not None:
                all_data['envelope_contexts'][name] = envelope_context.cpu().numpy()
            else:
                all_data['envelope_contexts'][name] = None
            
            # Get model outputs
            try:
                outputs = model(eegs.to(device))
            except Exception as ex:
                try:
                    outputs = model(eegs.to(device), envelope_context.to(device))
                except Exception as ex2:
                    print(f"Error processing model {name}: {ex2}")
                    continue
            
            # Store full outputs dictionary
            all_data['outputs'][name] = {}
            for key, value in outputs.items():
                all_data['outputs'][name][key] = value.cpu().numpy()
    
    # Save data using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Saved model data to {save_path}")

def main(base_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data",
        test_filelist = "filelists/filelist_3s_80pct_test_no_overlap/test_no_overlap_3s.csv"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    checkpoint_dir = "checkpoints"
    filtering_type = "none"
    
    transform_eeg = StandardizeEEG()
    transform_envelope = StandardizeEnvelope(
        filter_type=filtering_type,
        fs=64
    )
    
    # Create visualization directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Default context and prediction seconds
    default_context_seconds = 1.5
    default_prediction_seconds = 3.5
    
    # Model mappings with potential context information
    model_mappings = {
        "baselinedecoder2": "baseline_decoder2",
        "vlaai": "vlaai",
        "happyquoka4": "happyquoka4",
        "happyquokascaled": "happyquoka_scaled",
        "twobranchcontext0.05secdynamic": "two_branch_context_0.05sec_dynamic",
        "twobranchcontext0.1secdynamic": "two_branch_context_0.1sec_dynamic",
        "twobranchcontext0.15secdynamic": "two_branch_context_0.15sec_dynamic",
        "twobranchcontext0.2secdynamic": "two_branch_context_0.2sec_dynamic",
        "twobranchcontext0.25secdynamic": "two_branch_context_0.25sec_dynamic",
        "twobranchcontext0.5secdynamic": "two_branch_context_0.5sec_dynamic",
        "twobranchcontext1.0secdynamic": "two_branch_context_1.0sec_dynamic",
        "twobranchcontext1.5secdynamic": "two_branch_context_1.5sec_dynamic",
        "twobranchfrozeneegcontext0.0secdynamic": "two_branch_frozen_eeg_context_0.0sec_dynamic",
        "twobranchfrozeneegcontext0.05secenvelopeonly": "two_branch_frozen_eeg_context_0.05sec_envelope_only",
        "twobranchfrozeneegcontext0.1secenvelopeonly": "two_branch_frozen_eeg_context_0.1sec_envelope_only",
        "twobranchfrozeneegcontext0.15secenvelopeonly": "two_branch_frozen_eeg_context_0.15sec_envelope_only",
        "twobranchfrozeneegcontext0.2secenvelopeonly": "two_branch_frozen_eeg_context_0.2sec_envelope_only",
        "twobranchfrozeneegcontext0.25secenvelopeonly": "two_branch_frozen_eeg_context_0.25sec_envelope_only",
        "twobranchfrozeneegcontext0.5secenvelopeonly": "two_branch_frozen_eeg_context_0.5sec_envelope_only",
        "twobranchfrozeneegcontext1.0secenvelopeonly": "two_branch_frozen_eeg_context_1.0sec_envelope_only",
        "twobranchfrozeneegcontext1.5secenvelopeonly": "two_branch_frozen_eeg_context_1.5sec_envelope_only",
    }
    
    """{
        "baselinedecoder2": "baseline_decoder2",
        "vlaai": "vlaai",
        "happyquoka4": "happyquoka4",
        "twobranchfrozeneegcontext0.0secdynamic": "two_branch_frozen_eeg_context_0.0sec_dynamic",
        "twobranchfrozeneegcontext0.05secdynamic": "two_branch_frozen_eeg_context_0.05sec_dynamic",
        "twobranchfrozeneegcontext0.05secenvelopeonly": "two_branch_frozen_eeg_context_0.05sec_envelope_only",
        "twobranchfrozeneegcontext0.1secdynamic": "two_branch_frozen_eeg_context_0.1sec_dynamic",
        "twobranchfrozeneegcontext0.1secenvelopeonly": "two_branch_frozen_eeg_context_0.1sec_envelope_only",
        "twobranchfrozeneegcontext0.15secdynamic": "two_branch_frozen_eeg_context_0.15sec_dynamic",
        "twobranchfrozeneegcontext0.15secenvelopeonly": "two_branch_frozen_eeg_context_0.15sec_envelope_only",
        "twobranchfrozeneegcontext0.2secdynamic": "two_branch_frozen_eeg_context_0.2sec_dynamic",
        "twobranchfrozeneegcontext0.2secenvelopeonly": "two_branch_frozen_eeg_context_0.2sec_envelope_only",
        "twobranchfrozeneegcontext0.25secdynamic": "two_branch_frozen_eeg_context_0.25sec_dynamic",
        "twobranchfrozeneegcontext0.25secenvelopeonly": "two_branch_frozen_eeg_context_0.25sec_envelope_only",
        "twobranchfrozeneegcontext0.5secdynamic": "two_branch_frozen_eeg_context_0.5sec_dynamic",
        "twobranchfrozeneegcontext0.5secenvelopeonly": "two_branch_frozen_eeg_context_0.5sec_envelope_only",
        "twobranchfrozeneegcontext1.0secdynamic": "two_branch_frozen_eeg_context_1.0sec_dynamic",
        "twobranchfrozeneegcontext1.0secenvelopeonly": "two_branch_frozen_eeg_context_1.0sec_envelope_only",
        "twobranchfrozeneegcontext1.5secdynamic": "two_branch_frozen_eeg_context_1.5sec_dynamic",
        "twobranchfrozeneegcontext1.5secenvelopeonly": "two_branch_frozen_eeg_context_1.5sec_envelope_only",
        
    }"""
    
    # Discover and load all models
    print("\nDiscovering models...")
    models_info = discover_models(checkpoint_dir, model_mappings)
    if not models_info:
        print("No models found in the checkpoint directory!")
        return

    print(f"\nFound {len(models_info)} models:")
    for info in models_info:
        print(f"- {info['name']} (Directory: {info['dir_type']}, Registry: {info['type']}, Loss: {info['loss']})")
    
    # Create a dictionary to store model-specific dataloaders
    model_dataloaders = {}
    
    # Create dataloaders for each model based on context information
    for info in models_info:
        model_name = info['name']
        model_type = info['type']
        
        # Check if "context" is in the model type or name
        context_seconds = default_context_seconds
        prediction_seconds = default_prediction_seconds
        
        # Extract context seconds from model type if it contains "context"
        if "context" in model_type.lower() or "context" in info['dir_type'].lower():
            # Try to extract context seconds from the model type
            try:
                # Look for patterns like "context1.5sec" or "context_1.5sec"
                context_pattern = re.search(r'context[_]?(\d+\.?\d*)sec', model_type.lower())
                if not context_pattern:
                    context_pattern = re.search(r'context[_]?(\d+\.?\d*)sec', info['dir_type'].lower())
                
                if context_pattern:
                    extracted_context = float(context_pattern.group(1))
                    context_seconds = extracted_context
                    print(f"Extracted context seconds for {model_name}: {context_seconds}")
            except Exception as e:
                print(f"Error extracting context seconds for {model_name}: {str(e)}")
        
        # Create model-specific dataloader
        print(f"Creating dataloader for {model_name} with context_seconds={context_seconds}, prediction_seconds={prediction_seconds}")
        
        # Add context and prediction seconds to model info
        for i, info_dict in enumerate(models_info):
            if info_dict['name'] == model_name:
                models_info[i]['context_seconds'] = context_seconds
                models_info[i]['prediction_seconds'] = prediction_seconds
                break
        
        model_dataloaders[model_name] = {
            'vis': create_dataloader(
                filelist_path=test_filelist,
                base_dir=base_dir,
                batch_size=10,
                shuffle=True,
                transform_eeg=transform_eeg,
                transform_envelope=transform_envelope,
                num_workers=0,
                context_seconds=context_seconds,
                prediction_seconds=prediction_seconds
            ),
            'stats': create_dataloader(
                filelist_path=test_filelist,
                base_dir=base_dir,
                batch_size=2056,
                shuffle=False,
                transform_eeg=transform_eeg,
                transform_envelope=transform_envelope,
                num_workers=0,
                context_seconds=context_seconds,
                prediction_seconds=prediction_seconds
            ),
            'data': create_dataloader(  # New dataloader for saving full model data
                filelist_path=test_filelist,
                base_dir=base_dir,
                batch_size=2056,  # Smaller batch size to avoid memory issues
                shuffle=False,
                transform_eeg=transform_eeg,
                transform_envelope=transform_envelope,
                num_workers=0,
                context_seconds=context_seconds,
                prediction_seconds=prediction_seconds
            )
        }
    
    # After creating all dataloaders, print a summary
    print("\nModel-specific dataloader configuration:")
    print("=" * 80)
    for info in models_info:
        model_name = info['name']
        context_seconds = info.get('context_seconds', default_context_seconds)
        prediction_seconds = info.get('prediction_seconds', default_prediction_seconds)
        print(f"- {model_name}: context={context_seconds}s, prediction={prediction_seconds}s")
    print("=" * 80)
    
    # Also create default dataloaders for general use
    default_vis_loader = create_dataloader(
        filelist_path=test_filelist,
        base_dir=base_dir,
        batch_size=10,
        shuffle=True,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        num_workers=0,
        context_seconds=default_context_seconds,
        prediction_seconds=default_prediction_seconds
    )
    
    default_stats_loader = create_dataloader(
        filelist_path=test_filelist,
        base_dir=base_dir,
        batch_size=64,
        shuffle=False,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        num_workers=0,
        context_seconds=default_context_seconds,
        prediction_seconds=default_prediction_seconds
    )
    
    default_data_loader = create_dataloader(  # New default dataloader for saving full model data
        filelist_path=test_filelist,
        base_dir=base_dir,
        batch_size=64,
        shuffle=False,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        num_workers=0,
        context_seconds=default_context_seconds,
        prediction_seconds=default_prediction_seconds
    )
    
    # Load all models
    models = load_all_models(models_info, device)
    if not models:
        print("No models were successfully loaded. Exiting.")
        return
    
    # Filter models as needed
    allowed_model_names = set(info['name'] for info in models_info)
    models = {name: model for name, model in models.items() if name in allowed_model_names}
    
    # Get a batch for visualization
    print("\nCreating visualizations for 10 examples...")
    
    # Create visualizations by model type based on discovered models_info
    base_models = set(info['type'] for info in models_info)
    for base_model in base_models:
        # Filter models of this type
        model_subset = {
            name: model for name, model in models.items() 
            if any(info['type'] == base_model and info['name'] == name for info in models_info)
        }
        
        if model_subset:
            print(f"\nCreating visualizations for {base_model} models...")
            
            # Create model-specific test batches
            model_specific_batches = {}
            for model_name in model_subset.keys():
                # Use model-specific dataloader if available, otherwise use default
                vis_loader = model_dataloaders.get(model_name, {}).get('vis', default_vis_loader)
                model_specific_batches[model_name] = next(iter(vis_loader))
            
            # Envelope predictions
            save_path = f"{SAVE_DIR}/model_predictions_{base_model}.png"
            visualize_predictions(model_subset, model_specific_batches, save_path, models_info)
            print(f"Saved envelope predictions to {save_path}")
    
    # Create visualizations by loss function
    loss_types = set(info['loss'] for info in models_info)
    for loss_type in loss_types:
        # Filter models with this loss
        model_subset = {
            name: model for name, model in models.items() 
            if any(info['loss'] == loss_type and info['name'] == name for info in models_info)
        }
        
        if model_subset:
            print(f"\nCreating visualizations for {loss_type} loss models...")
            
            # Create model-specific test batches
            model_specific_batches = {}
            for model_name in model_subset.keys():
                # Use model-specific dataloader if available, otherwise use default
                vis_loader = model_dataloaders.get(model_name, {}).get('vis', default_vis_loader)
                model_specific_batches[model_name] = next(iter(vis_loader))
            
            # Envelope predictions
            save_path = f"{SAVE_DIR}/model_predictions_loss_{loss_type}.png"
            visualize_predictions(model_subset, model_specific_batches, save_path, models_info)
            print(f"Saved envelope predictions to {save_path}")
    
    # Create visualization with all models together
    print("\nCreating visualizations with all models...")
    
    # Create model-specific test batches for all models
    all_model_batches = {}
    for model_name in models.keys():
        # Use model-specific dataloader if available, otherwise use default
        vis_loader = model_dataloaders.get(model_name, {}).get('vis', default_vis_loader)
        all_model_batches[model_name] = next(iter(vis_loader))
    
    save_path = f"{SAVE_DIR}/model_predictions_all.png"
    visualize_predictions(models, all_model_batches, save_path, models_info)
    print(f"Saved all model predictions to {save_path}")
    
    # NEW: Save model data for later analysis
    print("\nSaving model data for later analysis...")
    
    # Create model-specific test batches for all models
    all_model_data_batches = {}
    for model_name in models.keys():
        # Use model-specific dataloader if available, otherwise use default
        data_loader = model_dataloaders.get(model_name, {}).get('data', default_data_loader)
        all_model_data_batches[model_name] = next(iter(data_loader))
    
    # Save model data
    #save_path = f"{SAVE_DIR}/model_data.pkl"
    #save_model_data(models, all_model_data_batches, save_path, models_info)
    
    # Process all samples for detailed statistics
    print("\nProcessing all samples for detailed statistics...")
    all_correlations = {name: [] for name in models.keys()}
    all_metadata = []
    
    # New: Track branch correlations for contextual models
    contextual_models = [name for name in models.keys() if 'context' in name.lower()]
    branch_correlations = {}
    for model_name in contextual_models:
        branch_correlations[model_name] = {
            'final': [],
            'envelope_branch': [],
            'eeg_branch': []
        }
    
    # Process each model with its specific dataloader
    for model_name, model in models.items():
        print(f"\nProcessing statistics for model: {model_name}")
        
        # Use model-specific dataloader if available, otherwise use default
        stats_loader = model_dataloaders.get(model_name, {}).get('stats', default_stats_loader)
        model_total_samples = 0
        model_metadata = []
        valid_samples = 0
        
        # NEW: Create a list to store all data for this model
        model_all_data = {
            'metadata': [],
            'inputs': [],
            'ground_truths': [],
            'envelope_contexts': [],  # Added envelope_contexts
            'outputs': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(stats_loader):
                metadata, eegs, envelopes, eeg_context, envelope_context = batch
                model_total_samples += len(eegs)
                
                try:
                    outputs = model(eegs.to(device))
                except Exception as ex:
                    try:
                        outputs = model(eegs.to(device), envelope_context.to(device))
                    except Exception as ex2:
                        print(f"Error processing model {model_name} in detailed statistics: {ex2}")
                        continue
                
                # NEW: Store batch data
                for i in range(len(eegs)):
                    model_all_data['metadata'].append({
                        'subject': metadata['sub'][i],
                        'audio_file': metadata['audio_file'][i],
                        'paths_eeg': metadata['paths_eeg'][i],
                        'paths_envelope': metadata['paths_envelope'][i],
                        'start': metadata['start'][i].item(),
                        'end': metadata['end'][i].item()
                    })
                    model_all_data['inputs'].append(eegs[i].cpu().numpy())
                    model_all_data['ground_truths'].append(envelopes[i].cpu().numpy())
                    
                    # Store envelope context if available
                    if envelope_context is not None:
                        model_all_data['envelope_contexts'].append(envelope_context[i].cpu().numpy())
                    else:
                        model_all_data['envelope_contexts'].append(None)
                    
                    # Store full outputs dictionary for this sample
                    sample_outputs = {}
                    for key, value in outputs.items():
                        sample_outputs[key] = value[i].cpu().numpy()
                    model_all_data['outputs'].append(sample_outputs)
                
                # Get predictions from outputs dictionary
                pred = outputs['envelope']
                ground_truth = envelopes.numpy()
                
                # New: Extract branch outputs for contextual models
                if 'context' in model_name.lower():
                    envelope_branch_pred = None
                    eeg_branch_pred = None
                    
                    if 'envelope_branch' in outputs:
                        envelope_branch_pred = outputs['envelope_branch'].cpu().numpy()
                    if 'eeg_branch' in outputs:
                        eeg_branch_pred = outputs['eeg_branch'].cpu().numpy()
                
                # --- New cropping for correlation: only the last 10 - context seconds ---
                pred = pred.cpu().numpy()
                pred = pred[:, :CROP_END, :]
                ground_truth = ground_truth[:, :CROP_END, :]
                
                # Crop branch outputs if they exist
                if 'context' in model_name.lower():
                    if envelope_branch_pred is not None:
                        envelope_branch_pred = envelope_branch_pred[:, :CROP_END, :]
                    if eeg_branch_pred is not None:
                        eeg_branch_pred = eeg_branch_pred[:, :CROP_END, :]
                # ------------------------------------------------------------------
                
                # Calculate correlations for each sample in batch using the cropped signals
                for i in range(len(eegs)):
                    # Normalize signals
                    norm_gt = normalize_signal(ground_truth[i, :, 0])
                    norm_pred = normalize_signal(pred[i, :, 0])
                    
                    # Calculate correlation
                    try:
                        corr = np.corrcoef(norm_gt, norm_pred)[0, 1]
                        # Check if correlation is valid (not NaN and finite)
                        if np.isfinite(corr) and not np.isnan(corr):
                            all_correlations[model_name].append(corr)
                            
                            # Store metadata only for valid correlations
                            if len(model_metadata) < model_total_samples:
                                model_metadata.append({
                                    'subject': metadata['sub'][i],
                                    'audio_file': metadata['audio_file'][i],
                                    'paths_eeg': metadata['paths_eeg'][i],
                                    'paths_envelope': metadata['paths_envelope'][i],
                                    'start': metadata['start'][i].item(),
                                    'end': metadata['end'][i].item()
                                })
                            valid_samples += 1
                            
                            # New: Store branch correlations for contextual models
                            if 'context' in model_name.lower():
                                branch_correlations[model_name]['final'].append(corr)
                                
                                # Calculate envelope branch correlation if available
                                if envelope_branch_pred is not None:
                                    norm_env_branch = normalize_signal(envelope_branch_pred[i, :, 0])
                                    env_branch_corr = np.corrcoef(norm_gt, norm_env_branch)[0, 1]
                                    if np.isfinite(env_branch_corr) and not np.isnan(env_branch_corr):
                                        branch_correlations[model_name]['envelope_branch'].append(env_branch_corr)
                                    else:
                                        branch_correlations[model_name]['envelope_branch'].append(None)
                                else:
                                    branch_correlations[model_name]['envelope_branch'].append(None)
                                
                                # Calculate EEG branch correlation if available
                                if eeg_branch_pred is not None:
                                    norm_eeg_branch = normalize_signal(eeg_branch_pred[i, :, 0])
                                    eeg_branch_corr = np.corrcoef(norm_gt, norm_eeg_branch)[0, 1]
                                    if np.isfinite(eeg_branch_corr) and not np.isnan(eeg_branch_corr):
                                        branch_correlations[model_name]['eeg_branch'].append(eeg_branch_corr)
                                    else:
                                        branch_correlations[model_name]['eeg_branch'].append(None)
                                else:
                                    branch_correlations[model_name]['eeg_branch'].append(None)
                        else:
                            print(f"Warning: Invalid correlation for {model_name}, sample {i} in batch {batch_idx}")
                    except Exception as e:
                        print(f"Error calculating correlation for {model_name}, sample {i} in batch {batch_idx}: {str(e)}")
                
                print(f"Processed batch {batch_idx + 1} for {model_name}, valid samples: {valid_samples}/{model_total_samples}")
        
        print(f"Total valid samples for {model_name}: {valid_samples}/{model_total_samples}")
        
        # Store metadata for the first model only (but only for valid samples)
        if not all_metadata and model_metadata:
            all_metadata = model_metadata[:valid_samples]
        
        # NEW: Save all data for this model
        model_name_clean = model_name.replace('/', '_').replace('\\', '_')
        model_save_path = f"{SAVE_DIR}/model_data_{model_name_clean}.pkl"
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_all_data, f)
        print(f"Saved all data for model {model_name} to {model_save_path}")
    
    # Ensure all models have the same number of samples and align metadata
    min_samples = min(len(corrs) for corrs in all_correlations.values())
    all_correlations = {name: corrs[:min_samples] for name, corrs in all_correlations.items()}
    all_metadata = all_metadata[:min_samples]
    
    # Save detailed statistics to CSV
    stats_df = pd.DataFrame({
        'subject': [m['subject'] for m in all_metadata],
        'audio_file': [m['audio_file'] for m in all_metadata],
        'paths_eeg': [m['paths_eeg'] for m in all_metadata],
        'paths_envelope': [m['paths_envelope'] for m in all_metadata],
        'start': [m['start'] for m in all_metadata],
        'end': [m['end'] for m in all_metadata],
        **{name: corrs for name, corrs in all_correlations.items()}
    })
    
    # Save to CSV
    stats_df.to_csv(f'{SAVE_DIR}/detailed_statistics.csv', index=False)
    print(f"Saved detailed statistics to {SAVE_DIR}/detailed_statistics.csv")
    
    # New: Save branch correlations for contextual models
    if contextual_models:
        # Create a DataFrame for each contextual model
        for model_name in contextual_models:
            # Filter out None values and ensure all lists have the same length
            valid_indices = []
            for i in range(len(branch_correlations[model_name]['final'])):
                if (branch_correlations[model_name]['envelope_branch'][i] is not None or 
                    branch_correlations[model_name]['eeg_branch'][i] is not None):
                    valid_indices.append(i)
            
            if valid_indices:
                # Create DataFrame with valid indices
                branch_df = pd.DataFrame({
                    'subject': [all_metadata[i]['subject'] for i in valid_indices],
                    'audio_file': [all_metadata[i]['audio_file'] for i in valid_indices],
                    'paths_eeg': [all_metadata[i]['paths_eeg'] for i in valid_indices],
                    'paths_envelope': [all_metadata[i]['paths_envelope'] for i in valid_indices],
                    'start': [all_metadata[i]['start'] for i in valid_indices],
                    'end': [all_metadata[i]['end'] for i in valid_indices],
                    'final_correlation': [branch_correlations[model_name]['final'][i] for i in valid_indices],
                    'envelope_branch_correlation': [branch_correlations[model_name]['envelope_branch'][i] for i in valid_indices],
                    'eeg_branch_correlation': [branch_correlations[model_name]['eeg_branch'][i] for i in valid_indices]
                })
                
                # Save to CSV
                model_name_clean = model_name.replace('/', '_').replace('\\', '_')
                branch_df.to_csv(f'{SAVE_DIR}/branch_correlations_{model_name_clean}.csv', index=False)
                print(f"Saved branch correlations for {model_name} to {SAVE_DIR}/branch_correlations_{model_name_clean}.csv")
        
        # Create a summary DataFrame with mean correlations
        summary_data = []
        for model_name in contextual_models:
            # Calculate mean correlations, filtering out None values
            final_corrs = [c for c in branch_correlations[model_name]['final'] if c is not None]
            env_branch_corrs = [c for c in branch_correlations[model_name]['envelope_branch'] if c is not None]
            eeg_branch_corrs = [c for c in branch_correlations[model_name]['eeg_branch'] if c is not None]
            
            mean_final = np.mean(final_corrs) if final_corrs else np.nan
            mean_env_branch = np.mean(env_branch_corrs) if env_branch_corrs else np.nan
            mean_eeg_branch = np.mean(eeg_branch_corrs) if eeg_branch_corrs else np.nan
            
            summary_data.append({
                'model': model_name,
                'mean_final_correlation': mean_final,
                'mean_envelope_branch_correlation': mean_env_branch,
                'mean_eeg_branch_correlation': mean_eeg_branch,
                'num_samples': len(final_corrs)
            })
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{SAVE_DIR}/branch_correlations_summary.csv', index=False)
        print(f"Saved branch correlations summary to {SAVE_DIR}/branch_correlations_summary.csv")

    # Create violin plot with statistical significance
    create_violin_plot(stats_df, f'{SAVE_DIR}/model_comparison_violin.png', models_info)

if __name__ == "__main__":

    # Run out-of-distribution testing
    # DTU Single Speaker dataset (OOD)
    # base_dir = "/home/karan/sda_link/datasets/DTU_Single_Speaker"
    # test_filelist = "filelists/filelist_3s_80pct_test_no_overlap/DTU_no_overlap_3s.csv"
    
    # Set crop length to 1 second (64 samples at 64Hz)
    CROP_END = int(3 * 64)
    
    # Create output directory
    # SAVE_DIR = "model_comparison_analysis/visualizations_OOD_test_cropped_ending_at_0.5sec_context_0.5sec"
    # os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Run main analysis
    # main(base_dir, test_filelist)
    
    # ICASSP 2024 EEG dataset (in-distribution)
    base_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data"
    #test_filelist = "filelists/filelist_3s_80pct_test_no_overlap/test_no_overlap_3s.csv"
    test_filelist = "filelists/filelist_5s_80pct_test_no_overlap/test_no_overlap_5s.csv"
    
    # Create output directory
    SAVE_DIR = "checkpoints/model_comparison_analysis/test_visualizations_test_cropped_ending_at_0.5sec"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Run main analysis
    main(base_dir, test_filelist)