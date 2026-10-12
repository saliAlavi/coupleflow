import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import wilcoxon

# Directory where noise test results are stored
NOISE_TEST_DIR = "checkpoints/model_comparison_analysis/noise_robustness_results_unified"
OUTPUT_DIR = "checkpoints/model_comparison_analysis/noise_robustness_results_unified"

def load_all_results():
    """Load all noise test results from CSV files."""
    # Check if combined results file exists
    combined_path = os.path.join(NOISE_TEST_DIR, "noise_results_all_models.csv")
    if os.path.exists(combined_path):
        print(f"Loading combined results from {combined_path}")
        return pd.read_csv(combined_path)
    
    # Otherwise, load individual files and combine them
    print("Loading individual model results...")
    all_results = []
    
    for file in os.listdir(NOISE_TEST_DIR):
        if file.startswith("noise_results_") and file.endswith(".csv"):
            file_path = os.path.join(NOISE_TEST_DIR, file)
            try:
                df = pd.read_csv(file_path)
                all_results.append(df)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        raise FileNotFoundError("No noise test result files found")

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
        "two_branch_frozen_eeg_context_0.0sec_envelope_only": "HappyQuoka-TwoBranch-Noise-EnvelopeOnly",
        "two_branch_context_0.05sec_dynamic": "HappyQuoka-TwoBranch-Context-0.05s",
        "two_branch_frozen_eeg_context_0.05sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.05s-EnvelopeOnly",
        "two_branch_context_0.1sec_dynamic": "HappyQuoka-TwoBranch-Context-0.1s",
        "two_branch_frozen_eeg_context_0.1sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.1s-EnvelopeOnly",
        "two_branch_context_0.15sec_dynamic": "HappyQuoka-TwoBranch-Context-0.15s",
        "two_branch_frozen_eeg_context_0.15sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.15s-EnvelopeOnly",
        "two_branch_context_0.2sec_dynamic": "HappyQuoka-TwoBranch-Context-0.2s",
        "two_branch_frozen_eeg_context_0.2sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.2s-EnvelopeOnly",
        "two_branch_context_0.25sec_dynamic": "HappyQuoka-TwoBranch-Context-0.25s",
        "two_branch_frozen_eeg_context_0.25sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.25s-EnvelopeOnly",
        "two_branch_context_0.5sec_dynamic": "HappyQuoka-TwoBranch-Context-0.5s",
        "two_branch_frozen_eeg_context_0.5sec_envelope_only": "HappyQuoka-TwoBranch-Context-0.5s-EnvelopeOnly",
        "two_branch_context_1.0sec_dynamic": "HappyQuoka-TwoBranch-Context-1.0s",
        "two_branch_frozen_eeg_context_1.0sec_envelope_only": "HappyQuoka-TwoBranch-Context-1.0s-EnvelopeOnly",
        "two_branch_context_1.5sec_dynamic": "HappyQuoka-TwoBranch-Context-1.5s",
        "two_branch_frozen_eeg_context_1.5sec_envelope_only": "HappyQuoka-TwoBranch-Context-1.5s-EnvelopeOnly",
    }
    
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
    elif "two_branch" in model_name.lower() or "twobranch" in model_name.lower():
        # Extract context time if available
        context_match = re.search(r'context[_]?(\d+\.?\d*)sec', model_name.lower())
        if context_match:
            context_time = context_match.group(1)
            if context_time == "0.0":
                return "HappyQuoka-TwoBranch-Noise"
            if "envelope" in model_name.lower():
                return f"HappyQuoka-TwoBranch-Context-EnvelopeOnly-{context_time}s"
            else:
                return f"HappyQuoka-TwoBranch-Context-{context_time}s"
        else:
            return "TwoBranch"
    
    # If no pattern matches, return the original name
    return model_name

def get_custom_model_colors():
    """
    Create a custom color mapping for models with:
    - Non-TwoBranch models in shades of gray
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
        "HappyQuoka-TwoBranch-Context-0.05s-EnvelopeOnly": "#0000FF", # blue
        "HappyQuoka-TwoBranch-Context-0.1s": "#FFE0B2",  # Very light orange
        "HappyQuoka-TwoBranch-Context-0.1s-EnvelopeOnly": "#0000FF",  # blue
        "HappyQuoka-TwoBranch-Context-0.15s": "#FFCC80", # Light orange
        "HappyQuoka-TwoBranch-Context-0.15s-EnvelopeOnly": "#0000FF", # blue
        "HappyQuoka-TwoBranch-Context-0.2s": "#FFB74D",  # Medium light orange
        "HappyQuoka-TwoBranch-Context-0.2s-EnvelopeOnly": "#0000FF",  # blue
        "HappyQuoka-TwoBranch-Context-0.25s": "#FFA726", # Medium orange
        "HappyQuoka-TwoBranch-Context-0.25s-EnvelopeOnly": "#0000FF", # blue
        "HappyQuoka-TwoBranch-Context-0.5s": "#FF9800",  # Orange
        "HappyQuoka-TwoBranch-Context-0.5s-EnvelopeOnly": "#0000FF",  # blue
        "HappyQuoka-TwoBranch-Context-1.0s": "#F57C00",  # Medium dark orange
        "HappyQuoka-TwoBranch-Context-1.0s-EnvelopeOnly": "#0000FF",  # blue
        "HappyQuoka-TwoBranch-Context-1.5s": "#E65100",  # Dark orange
        "HappyQuoka-TwoBranch-Context-1.5s-EnvelopeOnly": "#0000FF",  # blue
    }
    
    return color_mapping

def create_noise_performance_curve_plot(df, save_path, noise_type=None):
    """
    Create a plot showing performance curves for all models across noise levels,
    with error bars showing standard error of the mean.
    
    Args:
        df: DataFrame with noise test results
        save_path: Path to save the plot
        noise_type: Type of noise to filter by (None for all)
    """
    # Create a copy of the dataframe with simplified model names
    df_copy = df.copy()
    df_copy['simple_model'] = df_copy['model'].apply(simplify_model_name)
    
    # Filter by noise type if specified
    if noise_type is not None:
        df_copy = df_copy[df_copy['noise_type'] == noise_type]
        if len(df_copy) == 0:
            print(f"No data found for noise type: {noise_type}")
            return
    
    # Get unique models
    models = df_copy['model'].unique()
    
    # Create a figure
    plt.figure(figsize=(14, 8))
    
    # Calculate mean and standard error for each model at each SNR level
    mean_data = df_copy.groupby(['model', 'simple_model', 'snr'])['correlation'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error of the mean
    mean_data['sem'] = mean_data['std'] / np.sqrt(mean_data['count'])
    
    # Create custom order for SNR levels with "Clean" first, then descending
    snr_levels = mean_data['snr'].unique()
    custom_order = ['Clean'] + sorted(
        [snr for snr in snr_levels if snr != 'Clean'],
        key=lambda x: float(x.split()[0]),
        reverse=True
    )
    
    # Get custom color mapping
    color_mapping = get_custom_model_colors()
    
    # Plot each model's performance curve with error bars
    for model in models:
        model_data = mean_data[mean_data['model'] == model]
        simple_model = model_data['simple_model'].iloc[0]  # Get simplified name
        
        # Sort by custom SNR order
        model_data['snr_order'] = model_data['snr'].apply(lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
        model_data = model_data.sort_values('snr_order')
        
        # Get color for this model
        color = color_mapping.get(simple_model, 'blue')  # Default to blue if not found
        
        # Plot the line with error bars
        plt.errorbar(
            model_data['snr'], 
            model_data['mean'], 
            yerr=model_data['sem'],
            marker='o', 
            linewidth=2, 
            label=simple_model, 
            color=color,
            capsize=4,
            elinewidth=1,
            markeredgewidth=1
        )
    
    # Customize the plot
    noise_title = f"{noise_type.capitalize()} Noise" if noise_type else "Noise"
    plt.title(f'Model Performance Across {noise_title} Levels', fontsize=16)
    plt.xlabel('SNR Level', fontsize=14)
    plt.ylabel('Correlation Coefficient (r)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis tick labels in the custom order
    plt.xticks(custom_order, rotation=45)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise performance curve plot to {save_path}")

def create_clean_performance_violin_plot(df, save_path, noise_type=None):
    """
    Create a violin plot showing model performance for the clean case only,
    with statistical significance tests between models.
    
    Args:
        df: DataFrame with noise test results
        save_path: Path to save the plot
        noise_type: Type of noise to filter by (None for all)
    """
    # Create a copy of the dataframe with simplified model names
    df_copy = df.copy()
    df_copy['simple_model'] = df_copy['model'].apply(simplify_model_name)
    
    # Filter by noise type if specified
    if noise_type is not None:
        df_copy = df_copy[df_copy['noise_type'] == noise_type]
        if len(df_copy) == 0:
            print(f"No data found for noise type: {noise_type}")
            return
    
    # Filter for clean data only
    clean_data = df_copy[df_copy['snr'] == 'Clean']
    
    # If no clean data is found, try using the highest SNR level
    if len(clean_data) == 0:
        # Get the highest SNR level
        snr_levels = df_copy['snr'].unique()
        numeric_snrs = [float(snr.split()[0]) for snr in snr_levels if snr != 'Clean']
        if numeric_snrs:
            highest_snr = f"{max(numeric_snrs)} dB"
            clean_data = df_copy[df_copy['snr'] == highest_snr]
            print(f"No 'Clean' data found, using highest SNR level: {highest_snr}")
    
    if len(clean_data) == 0:
        print("No clean or high SNR data found. Cannot create violin plot.")
        return
    
    # Create a pivot table with subjects as rows and models as columns
    pivot_data = clean_data.pivot_table(
        index='subject',
        columns='simple_model',
        values='correlation',
        aggfunc='mean'
    )
    
    # Get custom color mapping
    color_mapping = get_custom_model_colors()
    
    # Get model names in order of mean performance
    model_order = pivot_data.mean().sort_values().index.tolist()
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Create violin plot with custom colors
    ax = sns.violinplot(
        data=pivot_data[model_order], 
        inner="quartile", 
        palette={model: color_mapping.get(model, 'blue') for model in model_order}
    )
    
    # Perform Wilcoxon signed-rank test between consecutive models
    p_values = []
    for i in range(len(model_order) - 1):
        model_a = pivot_data[model_order[i]]
        model_b = pivot_data[model_order[i + 1]]
        
        # Filter out NaN values for the test
        valid_mask = ~(model_a.isna() | model_b.isna())
        if valid_mask.sum() >= 2:  # Need at least 2 pairs for the test
            stat, p_value = wilcoxon(model_a[valid_mask], model_b[valid_mask], alternative="two-sided")
            p_values.append(p_value)
        else:
            p_values.append(float('nan'))
    
    # Add p-values for significance
    for i, p_value in enumerate(p_values):
        if np.isfinite(p_value):
            x_position = i + 0.5  # Position between the two models
            y_position = max(pivot_data[model_order[i]].max(),
                            pivot_data[model_order[i + 1]].max()) + 0.02
            plt.text(x_position, y_position, f"p={p_value:.3g}", 
                    ha='center', fontsize=10, color='blue')
    
    # Customize the plot
    noise_title = f"{noise_type.capitalize()} Noise" if noise_type else "Noise"
    plt.title(f"Model Performance Comparison (Clean EEG, {noise_title} Test)", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Correlation Coefficient (r)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean performance values
    for i, model_name in enumerate(model_order):
        mean_perf = pivot_data[model_name].mean()
        plt.text(i, pivot_data.min().min() - 0.05, 
                f"μ={mean_perf:.3f}", 
                ha='center', va='top', 
                fontsize=10, color='darkred')
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved clean performance violin plot to {save_path}")

def create_noise_type_comparison_plot(df, save_path, snr_level=None):
    """
    Create a plot comparing model performance between EEG noise and context noise
    at a specific SNR level.
    
    Args:
        df: DataFrame with noise test results
        save_path: Path to save the plot
        snr_level: SNR level to compare (default: 0 dB)
    """
    # Create a copy of the dataframe with simplified model names
    df_copy = df.copy()
    df_copy['simple_model'] = df_copy['model'].apply(simplify_model_name)
    
    # Filter for specific SNR level
    if snr_level is None:
        # Default to 0 dB if available, otherwise use the middle SNR level
        snr_levels = sorted([snr for snr in df_copy['snr'].unique() if snr != 'Clean'], 
                           key=lambda x: float(x.split()[0]) if x != 'Clean' else 0)
        if '0 dB' in snr_levels:
            snr_level = '0 dB'
        elif len(snr_levels) > 0:
            snr_level = snr_levels[len(snr_levels) // 2]
        else:
            print("No SNR levels found for comparison.")
            return
    
    # Filter data for the selected SNR level
    snr_data = df_copy[df_copy['snr'] == snr_level]
    
    if len(snr_data) == 0:
        print(f"No data found for SNR level: {snr_level}")
        return
    
    # Filter for models that have both EEG and context noise results
    model_counts = snr_data.groupby('simple_model')['noise_type'].nunique()
    models_with_both = model_counts[model_counts >= 2].index.tolist()
    
    if not models_with_both:
        print("No models found with both EEG and context noise results.")
        return
    
    # Filter data for models with both noise types
    comparison_data = snr_data[snr_data['simple_model'].isin(models_with_both)]
    
    # Create a pivot table with models as rows and noise types as columns
    pivot_data = comparison_data.pivot_table(
        index=['simple_model', 'subject'],
        columns='noise_type',
        values='correlation',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate mean performance for each model and noise type
    model_means = pivot_data.groupby('simple_model')[['eeg', 'context']].mean()
    
    # Sort models by average performance across both noise types
    model_order = model_means.mean(axis=1).sort_values().index.tolist()
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Set up positions for grouped bars
    x = np.arange(len(model_order))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(x - width/2, model_means.loc[model_order, 'eeg'], width, label='EEG Noise', color='#4285F4')
    plt.bar(x + width/2, model_means.loc[model_order, 'context'], width, label='Context Noise', color='#EA4335')
    
    # Add error bars (standard error of the mean)
    for i, model in enumerate(model_order):
        model_data = pivot_data[pivot_data['simple_model'] == model]
        
        # Calculate standard error for EEG noise
        eeg_sem = model_data['eeg'].std() / np.sqrt(len(model_data))
        plt.errorbar(i - width/2, model_means.loc[model, 'eeg'], yerr=eeg_sem, fmt='none', color='black', capsize=3)
        
        # Calculate standard error for context noise
        context_sem = model_data['context'].std() / np.sqrt(len(model_data))
        plt.errorbar(i + width/2, model_means.loc[model, 'context'], yerr=context_sem, fmt='none', color='black', capsize=3)
        
        # Perform Wilcoxon signed-rank test between EEG and context noise
        valid_mask = ~(model_data['eeg'].isna() | model_data['context'].isna())
        if valid_mask.sum() >= 2:
            stat, p_value = wilcoxon(model_data['eeg'][valid_mask], model_data['context'][valid_mask], alternative="two-sided")
            
            # Add significance indicator
            if p_value < 0.05:
                y_pos = max(model_means.loc[model, 'eeg'], model_means.loc[model, 'context']) + 0.02
                plt.text(i, y_pos, f"p={p_value:.3g}*", ha='center', fontsize=9, color='blue')
            else:
                y_pos = max(model_means.loc[model, 'eeg'], model_means.loc[model, 'context']) + 0.02
                plt.text(i, y_pos, f"p={p_value:.3g}", ha='center', fontsize=9, color='blue')
    
    # Customize the plot
    plt.title(f'EEG vs. Context Noise Performance Comparison at {snr_level}', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Correlation Coefficient (r)', fontsize=14)
    plt.xticks(x, model_order, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise type comparison plot to {save_path}")

def main():
    """Main function to create noise performance plots."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all results
    try:
        results_df = load_all_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Check if the dataframe has a noise_type column
    if 'noise_type' not in results_df.columns:
        print("Warning: 'noise_type' column not found in results. Adding default 'eeg' noise type.")
        results_df['noise_type'] = 'eeg'
    
    # Create plots for all noise types combined
    create_noise_performance_curve_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'noise_performance_curves_all.png')
    )
    
    create_clean_performance_violin_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'clean_performance_violin_all.png')
    )
    
    # Create plots for EEG noise
    create_noise_performance_curve_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'noise_performance_curves_eeg.png'),
        noise_type='eeg'
    )
    
    create_clean_performance_violin_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'clean_performance_violin_eeg.png'),
        noise_type='eeg'
    )
    
    # Create plots for context noise
    create_noise_performance_curve_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'noise_performance_curves_context.png'),
        noise_type='context'
    )
    
    create_clean_performance_violin_plot(
        results_df,
        os.path.join(OUTPUT_DIR, 'clean_performance_violin_context.png'),
        noise_type='context'
    )
    
    # Create comparison plots between EEG and context noise
    for snr_level in ['20 dB', '10 dB', '5 dB', '0 dB', '-5 dB', '-10 dB', '-20 dB']:
        create_noise_type_comparison_plot(
            results_df,
            os.path.join(OUTPUT_DIR, f'noise_type_comparison_{snr_level.replace(" ", "_")}.png'),
            snr_level=snr_level
        )
    
    print("\nNoise performance plots created!")
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()