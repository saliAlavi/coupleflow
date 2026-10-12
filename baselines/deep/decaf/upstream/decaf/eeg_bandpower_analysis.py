import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
from scipy.signal import welch
from scipy.stats import pearsonr
from pathlib import Path

# Define output directory
OUTPUT_DIR = "checkpoints/model_comparison_analysis/bandpower_analysis"
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
        "TwoBranchFrozenEegContext0.0": "HappyQuoka-TwoBranch-Noise",
        "TwoBranchFrozenEegContext0.05": "HappyQuoka-TwoBranch-Context-0.05s",
        "TwoBranchFrozenEegContext0.1": "HappyQuoka-TwoBranch-Context-0.1s",
        "TwoBranchFrozenEegContext0.15": "HappyQuoka-TwoBranch-Context-0.15s",
        "TwoBranchFrozenEegContext0.2": "HappyQuoka-TwoBranch-Context-0.2s",
        "TwoBranchFrozenEegContext0.25": "HappyQuoka-TwoBranch-Context-0.25s",
        "TwoBranchFrozenEegContext0.5": "HappyQuoka-TwoBranch-Context-0.5s",
        "TwoBranchFrozenEegContext1.0": "HappyQuoka-TwoBranch-Context-1.0s",
        "TwoBranchFrozenEegContext1.5": "HappyQuoka-TwoBranch-Context-1.5s",
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
    elif "twobranch" in model_name.lower() or "two_branch" in model_name.lower():
        # Extract context time if available
        context_match = re.search(r'context(\d+\.?\d*)', model_name.lower())
        if context_match:
            context_time = context_match.group(1)
            if context_time == "0.0" or context_time == "0":
                return "HappyQuoka-TwoBranch-Noise"
            elif "envelope" in model_name.lower():
                return f"HappyQuoka-TwoBranch-Context-EnvelopeOnly-{context_time}s"
            else:
                return f"HappyQuoka-TwoBranch-Context-{context_time}s"
        else:
            return "TwoBranch"
    
    # If no pattern matches, return the original name
    return model_name

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

def zscore(data):
    """
    Z-score normalize the data.
    
    Args:
        data: NumPy array
        
    Returns:
        Z-score normalized data
    """
    # Add small epsilon to avoid divide by zero
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - np.mean(data)) / std

def compute_bandpower(eeg_chunk, sf=64, bands=None):
    """
    Compute bandpower for each band in 'bands' by averaging across all channels.
    
    Args:
        eeg_chunk: NumPy array of shape (T, C).
        sf: sampling frequency.
        bands: dict, e.g. {'delta': (0.5, 4), 'theta': (4, 8), ...}
        
    Returns:
        dict of average bandpower across channels for each band.
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta':  (12, 32),
            'gamma': (32, 64),
        }
    
    # welch returns PSD for each channel if axis=0 => shape: (n_freqs, C)
    #normalize the eeg chunk by removing the mean and dividing by the std
    eeg_chunk = (eeg_chunk - np.mean(eeg_chunk, axis=0)) / np.std(eeg_chunk, axis=0)
    freqs, psd = welch(eeg_chunk, sf, axis=0, nperseg=64)  
    # psd[i, ch] is the power spectral density at frequency freqs[i] for channel ch
    
    bandpowers = {}
    for band_name, (f_low, f_high) in bands.items():
        idx = np.logical_and(freqs >= f_low, freqs <= f_high)
        # Average the PSD in this frequency band, then average across channels
        band_avg = psd[idx, :].mean(axis=0)  # mean over freq, keep channels
        bandpowers[band_name] = band_avg.mean()  # mean across channels
    return bandpowers

def band_power_analysis(two_branch_path, happyquoka_path):
    """
    Analyze the relationship between EEG band power and model performance.
    
    Args:
        two_branch_path: Path to the two-branch model data file
        happyquoka_path: Path to the HappyQuoka model data file
        
    Returns:
        DataFrame with subject-level results
    """
    try:
        with open(two_branch_path, 'rb') as f:
            data_two_branch = pickle.load(f)

        with open(happyquoka_path, 'rb') as f:
            data_happyquoka4 = pickle.load(f)
    except Exception as e:
        print(f"Error loading model data: {e}")
        return None

    # Initialize lists to store data for the table
    metadata_list = []

    for i in range(len(data_two_branch['outputs'])):
        try:
            meta_two_branch = data_two_branch['metadata'][i]
            meta_happyquoka4 = data_happyquoka4['metadata'][i]
            
            # Verify that we're comparing the same samples
            if (meta_two_branch['subject'] != meta_happyquoka4['subject'] or
                meta_two_branch['audio_file'] != meta_happyquoka4['audio_file'] or
                meta_two_branch['start'] != meta_happyquoka4['start'] or
                meta_two_branch['end'] != meta_happyquoka4['end']):
                print(f"Warning: Metadata mismatch at index {i}, skipping")
                continue

            # Get ground truth and model outputs
            gt_data = data_two_branch['ground_truths'][i][:64].flatten()
            two_branch_output_data = data_two_branch['outputs'][i]["envelope"][:64].flatten()
            happyquoka4_output_data = data_happyquoka4['outputs'][i]["envelope"][:64].flatten()
            
            # Skip if any data is all zeros or NaN
            if (np.all(gt_data == 0) or np.all(two_branch_output_data == 0) or 
                np.all(happyquoka4_output_data == 0) or
                np.isnan(gt_data).any() or np.isnan(two_branch_output_data).any() or 
                np.isnan(happyquoka4_output_data).any()):
                print(f"Warning: Zero or NaN data at index {i}, skipping")
                continue
            
            # Z-score normalize
            gt = zscore(gt_data)
            two_branch_output = zscore(two_branch_output_data)
            happyquoka4_output = zscore(happyquoka4_output_data)

            # Calculate correlations
            corr_two_branch = np.corrcoef(two_branch_output, gt)[0, 1]
            corr_happyquoka4 = np.corrcoef(happyquoka4_output, gt)[0, 1]
            
            # Skip if correlations are NaN
            if np.isnan(corr_two_branch) or np.isnan(corr_happyquoka4):
                print(f"Warning: NaN correlation at index {i}, skipping")
                continue
                
            corr_diff = corr_two_branch - corr_happyquoka4

            # Compute bandpowers
            bandpowers = compute_bandpower(data_two_branch['inputs'][i])

            # Add data to metadata
            meta_entry = meta_two_branch.copy()
            meta_entry["corr_diff"] = corr_diff
            meta_entry["delta_power"] = bandpowers["delta"]
            meta_entry["theta_power"] = bandpowers["theta"]
            meta_entry["alpha_power"] = bandpowers["alpha"]
            meta_entry["beta_power"] = bandpowers["beta"]
            meta_entry["gamma_power"] = bandpowers["gamma"]
            meta_entry["corr_two_branch"] = corr_two_branch
            meta_entry["corr_happyquoka4"] = corr_happyquoka4

            # Store metadata
            metadata_list.append(meta_entry)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Create a DataFrame to store the results
    if not metadata_list:
        print("No valid data to analyze")
        return None
        
    results_df = pd.DataFrame(metadata_list)
    results_df.dropna(inplace=True)
    
    # Group by subject
    subject_results = results_df.groupby('subject', as_index=False).mean()
    return subject_results

def create_band_correlation_table(model_results):
    """
    Create a table showing correlations between band power and model performance.
    
    Args:
        model_results: Dictionary mapping model names to DataFrames with results
        
    Returns:
        DataFrame with correlation table
    """
    # Define the frequency bands
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Initialize lists to store table rows
    table_rows = []
    
    # For each model, calculate correlations with each band
    for model_name, results_df in model_results.items():
        row_data = {'Model': simplify_model_name(model_name)}
        
        for band in bands:
            band_col = f"{band}_power"
            if band_col in results_df.columns and 'corr_diff' in results_df.columns:
                # Calculate correlation between band power and correlation difference
                x = results_df[band_col].values
                y = results_df['corr_diff'].values
                
                # Skip if there's not enough data
                if len(x) < 2 or len(y) < 2:
                    row_data[f"{band}_corr"] = np.nan
                    row_data[f"{band}_pval"] = np.nan
                    continue
                
                r_val, p_val = pearsonr(x, y)
                
                # Store correlation and p-value
                row_data[f"{band}_corr"] = r_val
                row_data[f"{band}_pval"] = p_val
                row_data["mean_corr_diff"] = results_df["corr_diff"].mean()
        
        table_rows.append(row_data)
    
    # Create DataFrame
    table_df = pd.DataFrame(table_rows)
    
    # Sort by context time
    table_df['context_time'] = table_df['Model'].apply(extract_context_time)
    table_df = table_df.sort_values('context_time')
    table_df = table_df.drop('context_time', axis=1)
    
    return table_df

def format_band_correlation_table_with_stars(table_df):
    """
    Format the correlation table with significance stars.
    
    Args:
        table_df: DataFrame with correlation results
        
    Returns:
        Formatted DataFrame
    """
    formatted_df = pd.DataFrame()
    formatted_df['Model'] = table_df['Model']
    formatted_df['Mean Corr Diff'] = table_df['mean_corr_diff'].apply(lambda x: f"{x:.3f}")
    
    # Define the frequency bands
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Format correlations with significance stars
    for band in bands:
        corr_col = f"{band}_corr"
        pval_col = f"{band}_pval"
        
        if corr_col in table_df.columns and pval_col in table_df.columns:
            # Create a new column with formatted correlation values and significance stars
            formatted_df[band.capitalize()] = table_df.apply(
                lambda row: format_correlation_with_stars(row[corr_col], row[pval_col]), 
                axis=1
            )
    
    return formatted_df

def format_correlation_with_stars(corr, pval):
    """
    Format correlation with significance stars:
    *** for p < 0.001
    ** for p < 0.01
    * for p < 0.05
    
    Args:
        corr: Correlation value
        pval: p-value
        
    Returns:
        Formatted string
    """
    if pd.isna(corr) or pd.isna(pval):
        return "N/A"
    
    # Round correlation to 2 decimal places
    corr_str = f"{corr:.2f}"
    
    # Add significance stars
    if pval < 0.001:
        return f"{corr_str}***"
    elif pval < 0.01:
        return f"{corr_str}**"
    elif pval < 0.05:
        return f"{corr_str}*"
    else:
        return f"{corr_str}"

def create_heatmap(table_df, save_path):
    """
    Create a heatmap visualization of the correlation table.
    
    Args:
        table_df: DataFrame with correlation results
        save_path: Path to save the heatmap
    """
    # Extract correlation values
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    corr_cols = [f"{band}_corr" for band in bands]
    
    # Check if we have enough data for a heatmap
    if len(table_df) < 2:
        print("Not enough data for a heatmap (need at least 2 models)")
        return
    
    # Create a new DataFrame for the heatmap
    heatmap_data = pd.DataFrame(index=table_df['Model'])
    
    # Add each band's correlation to the heatmap data
    for band in bands:
        col = f"{band}_corr"
        if col in table_df.columns:
            heatmap_data[band.capitalize()] = table_df.set_index('Model')[col]
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='RdBu_r',
        center=0,
        vmin=-0.8,
        vmax=0.8,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Correlation (r)'}
    )
    
    # Add title and labels
    plt.title('Correlation Between EEG Band Power and Model Performance Improvement', fontsize=16)
    plt.ylabel('Model', fontsize=14)
    plt.xlabel('Frequency Band', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")

def main():
    """Main function to analyze EEG band power and model performance."""
    # Find model data files
    model_files = find_model_data_files(MODEL_DATA_DIR)
    
    if not model_files:
        print(f"No model data files found in {MODEL_DATA_DIR}")
        return
    
    # Check if HappyQuoka4 model is available
    happyquoka_key = None
    for key in model_files.keys():
        if "happyquoka" in key.lower():
            happyquoka_key = key
            break
    
    if not happyquoka_key:
        print("Error: HappyQuoka4 model not found. This analysis requires the HappyQuoka4 model as a baseline.")
        return
    
    happyquoka_path = model_files[happyquoka_key]
    print(f"Using {happyquoka_key} as baseline model: {os.path.basename(happyquoka_path)}")
    
    # Filter for TwoBranch models
    two_branch_models = {}
    for key, path in model_files.items():
        if "twobranch" in key.lower() or "two_branch" in key.lower():
            two_branch_models[key] = path
    
    if not two_branch_models:
        print("Error: No TwoBranch models found.")
        return
    
    print(f"Found {len(two_branch_models)} TwoBranch models:")
    for model_name, path in two_branch_models.items():
        print(f"- {model_name}: {os.path.basename(path)}")
    
    # Analyze band power for each TwoBranch model
    subject_results = {}
    for model_name, path in two_branch_models.items():
        print(f"\nAnalyzing {model_name}...")
        results = band_power_analysis(path, happyquoka_path)
        if results is not None:
            subject_results[model_name] = results
    
    if not subject_results:
        print("Error: No valid results to analyze.")
        return
    
    # Create correlation table
    print("\nCreating correlation table...")
    raw_table = create_band_correlation_table(subject_results)
    formatted_table = format_band_correlation_table_with_stars(raw_table)
    
    # Save tables to CSV
    raw_table.to_csv(os.path.join(OUTPUT_DIR, 'bandpower_correlation_raw.csv'))
    formatted_table.to_csv(os.path.join(OUTPUT_DIR, 'bandpower_correlation_formatted.csv'))
    print(f"Saved correlation tables to {OUTPUT_DIR}")
    
    # Create heatmap visualization
    create_heatmap(raw_table, os.path.join(OUTPUT_DIR, 'bandpower_correlation_heatmap.png'))
    
    # Display the formatted table
    print("\nBand Power Correlation Table:")
    print(formatted_table.to_string(index=False))
    
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()