import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
from pathlib import Path

# Define output directory
OUTPUT_DIR = "checkpoints/model_comparison_analysis/fusion_weights_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directory to search for model data files
MODEL_DATA_DIR = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec"

def find_model_data_files(directory):
    """
    Find all .pkl files in the specified directory that contain model data.
    
    Args:
        directory: Directory to search for .pkl files
        
    Returns:
        Dictionary mapping model names to file paths
    """
    model_files = {}
    
    # Find all .pkl files in the directory
    pkl_files = glob.glob(os.path.join(directory, "model_data_*.pkl"))
    
    for file_path in pkl_files:
        # Extract model name from filename
        filename = os.path.basename(file_path)
        match = re.search(r'model_data_([^_]+)', filename)
        if match:
            model_name = match.group(1)
            # Clean up model name
            model_name = model_name.replace("SecDynamic", "")
            model_files[model_name] = file_path
        else:
            print(f"Warning: Could not extract model name from {filename}")
    
    return model_files

def simplify_model_name(model_name):
    """
    Convert complex model names to simpler, more readable versions.
    
    Args:
        model_name: Original model name
        
    Returns:
        Simplified model name
    """
    # Map of original names to simplified names
    name_mapping = {
         "baseline_decoder2": "LinearBaseline",
        "vlaai": "VLAAI",
        "happyquoka4": "HappyQuoka",
        "two_branch_frozen_eeg_context_0.0sec_dynamic": "HappyQuoka-TwoBranch-Noise",
        
        "two_branch_context_0.05sec_dynamic": "HappyQuoka-TwoBranch-Context-0.05s",
        "two_branch_context_0.1sec_dynamic": "HappyQuoka-TwoBranch-Context-0.1s",
        "two_branch_context_0.15sec_dynamic": "HappyQuoka-TwoBranch-Context-0.15s",
        "two_branch_context_0.2sec_dynamic": "HappyQuoka-TwoBranch-Context-0.2s",
        "two_branch_context_0.25sec_dynamic": "HappyQuoka-TwoBranch-Context-0.25s",
        "two_branch_context_0.5sec_dynamic": "HappyQuoka-TwoBranch-Context-0.5s",
        "two_branch_context_1.0sec_dynamic": "HappyQuoka-TwoBranch-Context-1.0s",
        "two_branch_context_1.5sec_dynamic": "HappyQuoka-TwoBranch-Context-1.5s",
    }
    """
    "two_branch_frozen_eeg_context_0.0sec_envelope_only": "HappyQuoka-TwoBranch-Noise-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.05sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.05s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.1sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.1s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.15sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.15s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.2sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.2s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.25sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.25s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_0.5sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.5s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_1.0sec_envelope_only": "HappyQuoka-TwoBranch-Context-1.0s-EnvelopeOnly",
    "two_branch_frozen_eeg_context_1.5sec_envelope_only": "HappyQuoka-TwoBranch-Context-1.5s-EnvelopeOnly",
    """
    
    # Try direct mapping first
    if model_name in name_mapping:
        return name_mapping[model_name]
    
    # If not found, try to extract key components
    if "baseline" in model_name.lower():
        return "LinearBaseline"
    elif "vlaai" in model_name.lower():
        return "VLAAI"
    elif "happyquoka" in model_name.lower():
        return "HappyQuoka"
    elif "twobranch" in model_name.lower() or "two_branch" in model_name.lower():
        # Extract context time if available
        context_match = re.search(r'context(\d+\.?\d*)', model_name.lower())
        if context_match:
            context_time = context_match.group(1)
            if context_time == "0.0" or context_time == "0":
                if "envelopeonly" in model_name.lower():
                    return None
                else:
                    return "HappyQuoka-TwoBranch-Noise"
            else:
                if "envelopeonly" in model_name.lower():
                    return None
                else:
                    return f"HappyQuoka-TwoBranch-Context-{context_time}s"
        else:
            return "TwoBranch"
    
    # If no pattern matches, return the original name
    return model_name

def get_custom_model_colors():
    """
    Create a custom color mapping for models with:
    - TwoBranch-0s in black
    - Other TwoBranch models in shades of orange
    
    Returns:
        Dictionary mapping model names to colors
    """
    color_mapping = {
        # Non-TwoBranch models in shades of gray (light to dark)
        "LinearBaseline": "#D3D3D3",  # Light gray
        "VLAAI": "#A9A9A9",          # Medium gray
        "HappyQuoka": "#808080",     # Dark gray
        
        # TwoBranch-0s in black
        "HappyQuoka-TwoBranch-Noise": "#b26734",   # Brown
        
        # Context models in shades of orange (light to dark)
        "HappyQuoka-TwoBranch-Context-0.05s": "#FFF3E0", # Lightest orange
        "HappyQuoka-TwoBranch-Context-0.1s": "#FFE0B2",  # Very light orange
        "HappyQuoka-TwoBranch-Context-0.15s": "#FFCC80", # Light orange
        "HappyQuoka-TwoBranch-Context-0.2s": "#FFB74D",  # Medium light orange
        "HappyQuoka-TwoBranch-Context-0.25s": "#FFA726", # Medium orange
        "HappyQuoka-TwoBranch-Context-0.5s": "#FF9800",  # Orange
        "HappyQuoka-TwoBranch-Context-1.0s": "#F57C00",  # Medium dark orange
        "HappyQuoka-TwoBranch-Context-1.5s": "#E65100"   # Dark orange
    }
    
    return color_mapping

def extract_context_time(model_name):
    """
    Extract the context time from a model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Context time as a float, or -1 for non-context models
    """
    # For models with "Noise" in the name, return 0
    if "Noise" in model_name:
        return 0.0
    
    # Extract context time if available
    context_match = re.search(r'Context-(\d+\.?\d*)s', model_name)
    if context_match:
        return float(context_match.group(1))
    
    # For non-context models, return -1
    return -1.0

def load_fusion_weights(model_files):
    """
    Load fusion weights from all model data files.
    
    Args:
        model_files: Dictionary mapping model names to file paths
        
    Returns:
        Dictionary containing fusion weights data
    """
    eeg_means = []
    envelope_means = []
    simple_labels = []
    original_labels = []
    
    mean_eeg_weights_across_predictions = []
    mean_envelope_weights_across_predictions = []
    
    for model_name, path in model_files.items():
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            continue
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if the model has fusion weights
            if "outputs" not in model_data or not model_data["outputs"] or "fusion_weights" not in model_data["outputs"][0]:
                print(f"Warning: No fusion weights found in {path}")
                continue
            
            all_fusion_weights_eeg = [1 - i["fusion_weights"] for i in model_data["outputs"]]
            all_fusion_weights_envelope = [i["fusion_weights"] for i in model_data["outputs"]]
            
            eeg_mean = np.array(all_fusion_weights_eeg).reshape(-1).mean(axis=0)
            envelope_mean = np.array(all_fusion_weights_envelope).reshape(-1).mean(axis=0)

            mean_eeg_weights_across_predictions.append(np.array(all_fusion_weights_eeg).mean(axis=0))
            mean_envelope_weights_across_predictions.append(np.array(all_fusion_weights_envelope).mean(axis=0))
            
            eeg_means.append(eeg_mean)
            envelope_means.append(envelope_mean)
            
            # Store both original and simplified model names
            if (simplify_model_name(model_name) is not None):
                original_labels.append(model_name)
                simple_labels.append(simplify_model_name(model_name))
            
                print(f"Loaded fusion weights for {model_name}")
            else:
                print(f"Skipping {model_name} because it was not found in the name mapping")
        except Exception as e:
            print(f"Error loading or processing {path}: {str(e)}")
    
    return {
        'eeg_means': eeg_means,
        'envelope_means': envelope_means,
        'original_labels': original_labels,
        'simple_labels': simple_labels,
        'mean_eeg_weights': mean_eeg_weights_across_predictions,
        'mean_envelope_weights': mean_envelope_weights_across_predictions
    }

def create_fusion_weights_bar_plot(data, save_path):
    """
    Create a bar plot showing the mean fusion weights for EEG and envelope.
    
    Args:
        data: Dictionary containing fusion weights data
        save_path: Path to save the plot
    """
    eeg_means = data['eeg_means']
    envelope_means = data['envelope_means']
    simple_labels = data['simple_labels']
    
    # Sort data by context time
    sorted_indices = []
    for i, label in enumerate(simple_labels):
        # Extract context time
        context_time = extract_context_time(label)
        sorted_indices.append((i, context_time))
    
    # Sort by context time
    sorted_indices.sort(key=lambda x: x[1])
    sorted_indices = [i[0] for i in sorted_indices]
    
    # Reorder data
    eeg_means = [eeg_means[i] for i in sorted_indices]
    envelope_means = [envelope_means[i] for i in sorted_indices]
    simple_labels = [simple_labels[i] for i in sorted_indices]
    
    # Get custom colors
    color_mapping = get_custom_model_colors()
    colors = [color_mapping.get(label, '#808080') for label in simple_labels]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set up bar positions
    x = np.arange(len(simple_labels))
    width = 0.35
    
    # Create bars
    rects1 = ax.bar(x - width/2, eeg_means, width, label='EEG Weight', color='#4285F4', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, envelope_means, width, label='Envelope Weight', color='#EA4335', edgecolor='black', linewidth=1)
    
    # Add labels and title
    ax.set_ylabel('Mean Contribution Weight', fontsize=14)
    ax.set_title('EEG and Envelope Contribution Weights by Model', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(simple_labels, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved fusion weights bar plot to {save_path}")

def create_fusion_weights_line_plot(data, save_path):
    """
    Create a line plot showing the mean fusion weights across time steps.
    
    Args:
        data: Dictionary containing fusion weights data
        save_path: Path to save the plot
    """
    mean_eeg_weights = data['mean_eeg_weights']
    mean_envelope_weights = data['mean_envelope_weights']
    simple_labels = data['simple_labels']
    original_labels = data['original_labels']
    
    # Sort data by context time
    sorted_data = []
    for i, label in enumerate(simple_labels):
        # Extract context time
        context_time = extract_context_time(label)
        sorted_data.append((i, context_time, label))
    
    # Sort by context time
    sorted_data.sort(key=lambda x: x[1])
    
    # Get custom colors
    color_mapping = get_custom_model_colors()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each model's weights
    for i, context_time, label in sorted_data:
        color = color_mapping.get(label, '#808080')
        
        # Plot EEG weights
        time_steps = np.arange(len(mean_eeg_weights[i]))
        ax.plot(time_steps, mean_eeg_weights[i], label=f"{label} (EEG)", 
                color=color, linestyle='-', linewidth=2, marker='o', markersize=4)
        
        # Plot Envelope weights with lighter color
        lighter_color = color  # You could make this lighter if desired
        ax.plot(time_steps, mean_envelope_weights[i], label=f"{label} (Envelope)", 
                color=lighter_color, linestyle='--', linewidth=2, marker='s', markersize=4, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Mean Contribution Weight', fontsize=14)
    ax.set_title('EEG and Envelope Contribution Weights Across Time Steps', fontsize=16)
    
    # Add grid lines
    ax.grid(linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved fusion weights line plot to {save_path}")

def create_context_time_vs_weight_plot(data, save_path):
    """
    Create a scatter plot showing how fusion weights change with context time.
    
    Args:
        data: Dictionary containing fusion weights data
        save_path: Path to save the plot
    """
    eeg_means = data['eeg_means']
    envelope_means = data['envelope_means']
    simple_labels = data['simple_labels']
    
    # Extract context times and filter out non-context models
    context_data = []
    for i, label in enumerate(simple_labels):
        context_time = extract_context_time(label)
        if context_time >= 0:  # Include context models and noise model (0)
            context_data.append((context_time, eeg_means[i], envelope_means[i], label))
    
    # Sort by context time
    context_data.sort(key=lambda x: x[0])
    
    # Separate data
    context_times = [item[0] for item in context_data]
    eeg_weights = [item[1] for item in context_data]
    envelope_weights = [item[2] for item in context_data]
    labels = [item[3] for item in context_data]
    
    # Get custom colors
    color_mapping = get_custom_model_colors()
    colors = [color_mapping.get(label, '#808080') for label in labels]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot EEG weights
    ax.scatter(context_times, eeg_weights, label='EEG Weight', color='#4285F4', 
               s=100, marker='o', edgecolor='black', linewidth=1)
    ax.plot(context_times, eeg_weights, color='#4285F4', linestyle='-', linewidth=2)
    
    # Plot Envelope weights
    ax.scatter(context_times, envelope_weights, label='Envelope Weight', color='#EA4335', 
               s=100, marker='s', edgecolor='black', linewidth=1)
    ax.plot(context_times, envelope_weights, color='#EA4335', linestyle='-', linewidth=2)
    
    # Add labels for each point
    for i, (x, y_eeg, y_env, label) in enumerate(context_data):
        # Add label above the point
        if i % 2 == 0:  # Alternate above/below to avoid overlap
            ax.annotate(label, (x, y_eeg + 0.03), ha='center', va='bottom', fontsize=9)
        else:
            ax.annotate(label, (x, y_env - 0.03), ha='center', va='top', fontsize=9)
    
    # Add labels and title
    ax.set_xlabel('Context Time (seconds)', fontsize=14)
    ax.set_ylabel('Mean Contribution Weight', fontsize=14)
    ax.set_title('EEG and Envelope Contribution Weights vs. Context Time', fontsize=16)
    
    # Add grid lines
    ax.grid(linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Set x-axis to start at 0
    ax.set_xlim(-0.1, max(context_times) + 0.1)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved context time vs. weight plot to {save_path}")

def main():
    """Main function to create fusion weights plots."""
    # Find model data files
    model_files = find_model_data_files(MODEL_DATA_DIR)
    
    if not model_files:
        print(f"No model data files found in {MODEL_DATA_DIR}")
        return
    
    print(f"Found {len(model_files)} model data files:")
    for model_name, path in model_files.items():
        print(f"- {model_name}: {os.path.basename(path)}")
    
    # Load fusion weights data
    fusion_data = load_fusion_weights(model_files)
    
    if not fusion_data['simple_labels']:
        print("No fusion weights data loaded. Exiting.")
        return
    
    # Create bar plot
    create_fusion_weights_bar_plot(
        fusion_data,
        os.path.join(OUTPUT_DIR, 'fusion_weights_bar_plot.png')
    )
    
    # Create line plot if the weights are time series
    if any(len(weights) > 1 for weights in fusion_data['mean_eeg_weights']):
        create_fusion_weights_line_plot(
            fusion_data,
            os.path.join(OUTPUT_DIR, 'fusion_weights_line_plot.png')
        )
    
    # Create context time vs. weight plot
    create_context_time_vs_weight_plot(
        fusion_data,
        os.path.join(OUTPUT_DIR, 'context_time_vs_weight_plot.png')
    )
    
    print("\nFusion weights plots created!")
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()