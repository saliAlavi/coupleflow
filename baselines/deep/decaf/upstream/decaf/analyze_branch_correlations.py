import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import re
import glob
from pathlib import Path

# Define output directory
OUTPUT_DIR = "checkpoints/model_comparison_analysis/branch_correlation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directory to search for model data files
MODEL_DATA_DIR = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec"

def find_model_data_files(directory):
    """
    Find all .pkl files in the specified directory that contain model data.
    """
    model_files = {}
    pkl_files = glob.glob(os.path.join(directory, "model_data_*.pkl"))
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        match = re.search(r'model_data_([^_]+)', filename)
        if match:
            model_name = match.group(1)
            model_name = model_name.replace("SecDynamic", "")
            model_files[model_name] = file_path
        else:
            print(f"Warning: Could not extract model name from {filename}")
    
    return model_files

def simplify_model_name(model_name):
    """
    Convert complex model names to simpler, more readable versions.
    """
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
    
    if model_name in name_mapping:
        return name_mapping[model_name]
    
    if "baseline" in model_name.lower():
        return "LinearBaseline"
    elif "vlaai" in model_name.lower():
        return "VLAAI"
    elif "happyquoka" in model_name.lower():
        return "HappyQuoka"
    elif "twobranch" in model_name.lower() or "two_branch" in model_name.lower():
        context_match = re.search(r'context(\d+\.?\d*)', model_name.lower())
        if context_match:
            context_time = context_match.group(1)
            if context_time == "0.0" or context_time == "0":
                return "HappyQuoka-TwoBranch-Noise"
            else:
                return f"HappyQuoka-TwoBranch-Context-{context_time}s"
        else:
            return "TwoBranch"
    
    return model_name

def get_custom_model_colors():
    """
    Create a custom color mapping for models.
    """
    color_mapping = {
        "LinearBaseline": "#D3D3D3",
        "VLAAI": "#A9A9A9",
        "HappyQuoka": "#808080",
        "HappyQuoka-TwoBranch-Noise": "#b26734",
        "HappyQuoka-TwoBranch-Context-0.05s": "#FFF3E0",
        "HappyQuoka-TwoBranch-Context-0.1s": "#FFE0B2",
        "HappyQuoka-TwoBranch-Context-0.15s": "#FFCC80",
        "HappyQuoka-TwoBranch-Context-0.2s": "#FFB74D",
        "HappyQuoka-TwoBranch-Context-0.25s": "#FFA726",
        "HappyQuoka-TwoBranch-Context-0.5s": "#FF9800",
        "HappyQuoka-TwoBranch-Context-1.0s": "#F57C00",
        "HappyQuoka-TwoBranch-Context-1.5s": "#E65100"
    }
    return color_mapping

def calculate_branch_correlations(model_files):
    """
    Calculate correlations between branch outputs and ground truth for each model.
    """
    correlations = {
        'eeg_correlations': [],
        'envelope_correlations': [],
        'simple_labels': [],
        'original_labels': []
    }
    
    for model_name, path in model_files.items():
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            continue
            
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if model has the required data
            if ("outputs" not in model_data or 
                not model_data["outputs"] or 
                "eeg_branch" not in model_data["outputs"][0] or
                "envelope_branch" not in model_data["outputs"][0] or
                "ground_truth" not in model_data["outputs"][0]):
                print(f"Warning: Missing required data in {path}")
                continue
            
            # Calculate correlations for each output
            eeg_corrs = []
            env_corrs = []
            
            for output in model_data["outputs"]:
                eeg_pred = output["eeg_branch"]
                env_pred = output["envelope_branch"]
                ground_truth = output["ground_truth"]
                
                # Calculate Pearson correlation
                eeg_corr, _ = pearsonr(eeg_pred.flatten(), ground_truth.flatten())
                env_corr, _ = pearsonr(env_pred.flatten(), ground_truth.flatten())
                
                eeg_corrs.append(eeg_corr)
                env_corrs.append(env_corr)
            
            # Store mean correlations
            if (simplify_model_name(model_name) is not None):
                correlations['eeg_correlations'].append(np.mean(eeg_corrs))
                correlations['envelope_correlations'].append(np.mean(env_corrs))
                correlations['original_labels'].append(model_name)
                correlations['simple_labels'].append(simplify_model_name(model_name))
                
                print(f"Calculated correlations for {model_name}")
            else:
                print(f"Skipping {model_name} because it was not found in the name mapping")
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    return correlations

def create_correlation_bar_plot(data, save_path):
    """
    Create a bar plot showing the correlations for EEG and envelope branches.
    """
    eeg_corrs = data['eeg_correlations']
    env_corrs = data['envelope_correlations']
    simple_labels = data['simple_labels']
    
    # Sort by context time
    sorted_indices = []
    for i, label in enumerate(simple_labels):
        context_time = extract_context_time(label)
        sorted_indices.append((i, context_time))
    
    sorted_indices.sort(key=lambda x: x[1])
    sorted_indices = [i[0] for i in sorted_indices]
    
    # Reorder data
    eeg_corrs = [eeg_corrs[i] for i in sorted_indices]
    env_corrs = [env_corrs[i] for i in sorted_indices]
    simple_labels = [simple_labels[i] for i in sorted_indices]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(simple_labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, eeg_corrs, width, label='EEG Branch', 
                    color='#4285F4', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, env_corrs, width, label='Envelope Branch', 
                    color='#EA4335', edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Correlation with Ground Truth', fontsize=14)
    ax.set_title('Branch Correlations with Ground Truth by Model', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(simple_labels, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(-1, 1)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation bar plot to {save_path}")

def extract_context_time(model_name):
    """
    Extract the context time from a model name.
    """
    if "Noise" in model_name:
        return 0.0
    
    context_match = re.search(r'Context-(\d+\.?\d*)s', model_name)
    if context_match:
        return float(context_match.group(1))
    
    return -1.0

def create_context_vs_correlation_plot(data, save_path):
    """
    Create a scatter plot showing how correlations change with context time.
    """
    eeg_corrs = data['eeg_correlations']
    env_corrs = data['envelope_correlations']
    simple_labels = data['simple_labels']
    
    # Extract context times and filter data
    context_data = []
    for i, label in enumerate(simple_labels):
        context_time = extract_context_time(label)
        if context_time >= 0:
            context_data.append((context_time, eeg_corrs[i], env_corrs[i], label))
    
    context_data.sort(key=lambda x: x[0])
    
    context_times = [item[0] for item in context_data]
    eeg_corrs = [item[1] for item in context_data]
    env_corrs = [item[2] for item in context_data]
    labels = [item[3] for item in context_data]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.scatter(context_times, eeg_corrs, label='EEG Branch', color='#4285F4',
               s=100, marker='o', edgecolor='black', linewidth=1)
    ax.plot(context_times, eeg_corrs, color='#4285F4', linestyle='-', linewidth=2)
    
    ax.scatter(context_times, env_corrs, label='Envelope Branch', color='#EA4335',
               s=100, marker='s', edgecolor='black', linewidth=1)
    ax.plot(context_times, env_corrs, color='#EA4335', linestyle='-', linewidth=2)
    
    for i, (x, y_eeg, y_env, label) in enumerate(context_data):
        if i % 2 == 0:
            ax.annotate(label, (x, y_eeg + 0.03), ha='center', va='bottom', fontsize=9)
        else:
            ax.annotate(label, (x, y_env - 0.03), ha='center', va='top', fontsize=9)
    
    ax.set_xlabel('Context Time (seconds)', fontsize=14)
    ax.set_ylabel('Correlation with Ground Truth', fontsize=14)
    ax.set_title('Branch Correlations vs. Context Time', fontsize=16)
    
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.1, max(context_times) + 0.1)
    
    ax.legend(fontsize=12)
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved context vs. correlation plot to {save_path}")

def main():
    """Main function to create branch correlation plots."""
    model_files = find_model_data_files(MODEL_DATA_DIR)
    
    if not model_files:
        print(f"No model data files found in {MODEL_DATA_DIR}")
        return
    
    print(f"Found {len(model_files)} model data files:")
    for model_name, path in model_files.items():
        print(f"- {model_name}: {os.path.basename(path)}")
    
    correlation_data = calculate_branch_correlations(model_files)
    
    if not correlation_data['simple_labels']:
        print("No correlation data loaded. Exiting.")
        return
    
    create_correlation_bar_plot(
        correlation_data,
        os.path.join(OUTPUT_DIR, 'branch_correlations_bar_plot.png')
    )
    
    create_context_vs_correlation_plot(
        correlation_data,
        os.path.join(OUTPUT_DIR, 'context_vs_correlation_plot.png')
    )
    
    print("\nBranch correlation plots created!")
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 