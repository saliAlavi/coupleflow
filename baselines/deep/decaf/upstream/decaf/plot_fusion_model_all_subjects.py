#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from organize_models import organize_model_paths
from tqdm import tqdm

def compute_psd_welch(list_of_arrays, sampling_rate):
    """
    Compute the power spectral density using Welch's method for a list of arrays.
    
    Args:
        list_of_arrays: List of arrays to compute PSD for
        sampling_rate: Sampling rate in Hz
    
    Returns:
        psd_list: List of PSDs
        freqs: Frequency bins
    """
    psd_list = []
    freqs_values = None
    
    for x in list_of_arrays:
        freqs, Pxx = signal.welch(x.reshape(-1), sampling_rate, nperseg=224, noverlap=32)
        if freqs_values is None:
            freqs_values = np.array(freqs)  # Convert to numpy array and store
        psd_list.append(Pxx)
    
    return psd_list, freqs_values

def filter_subject_data(data, subject_id):
    """
    Filter data for a specific subject.
    
    Args:
        data: Dictionary containing model data
        subject_id: Subject ID to filter for
    
    Returns:
        filtered_data: Dictionary with filtered data for the specified subject
    """
    filtered_indices = []
    for i, metadata in enumerate(data["metadata"]):
        if metadata["subject"] == subject_id:
            filtered_indices.append(i)
    
    filtered_data = {
        "metadata": [data["metadata"][i] for i in filtered_indices],
        "ground_truths": [data["ground_truths"][i] for i in filtered_indices],
        "inputs": [data["inputs"][i] for i in filtered_indices] if "inputs" in data else None,
        "outputs": [data["outputs"][i] for i in filtered_indices]
    }
    
    return filtered_data

def normalize(x):
    """Normalize data by z-score"""
    return (x - x.mean()) / x.std()

def safe_correlation(x, y):
    """
    Calculate correlation between two arrays, handling edge cases.
    Returns 0 if correlation cannot be calculated (e.g., all zeros in ground truth).
    
    Args:
        x: First array
        y: Second array
    
    Returns:
        correlation: Pearson correlation coefficient (0 if calculation fails)
    """
    # Check if ground truth (x) is all zeros
    if np.all(x == 0):
        return 0.0
        
    # Remove any NaN values from both arrays
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 2:  # Need at least 2 points for correlation
        return 0.0
        
    try:
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        return 0.0 if np.isnan(corr) else corr
    except:
        return 0.0

def filter_signal_to_bands(signal_list, sampling_rate=64):
    """
    Filter signals into different frequency bands.
    
    Args:
        signal_list: List of signals to filter
        sampling_rate: Sampling rate in Hz (default: 64)
    
    Returns:
        filtered_signals: Dictionary with filtered signals for each band
    """
    # Define frequency bands (in Hz)
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 32)  # Limited by Nyquist frequency (sampling_rate/2)
    }
    
    filtered_signals = {band: [] for band in bands}
    
    for signal_data in signal_list:
        signal_data = signal_data.reshape(-1)
        
        for band, (low, high) in bands.items():
            # Design bandpass filter
            nyq = 0.5 * sampling_rate
            
            # Ensure frequencies are within valid range (0 < f < 1)
            low_norm = max(0.001, low / nyq)  # Ensure it's greater than 0
            high_norm = min(0.999, high / nyq)  # Ensure it's less than 1
            
            # Use a higher order filter for narrow bands
            if high - low < 5:
                order = 6
            else:
                order = 4
                
            b, a = signal.butter(order, [low_norm, high_norm], btype='band')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, signal_data)
            filtered_signals[band].append(filtered)
    
    return filtered_signals

def analyze_fusion_model_for_all_subjects():
    """Analyze fusion model performance across all subjects"""
    
    # List of subjects to process (sub-001 to sub-085)
    all_subjects = [f"sub-{i:03d}" for i in range(1, 86)]
    
    # Create output directory structure
    base_output_dir = "plots_fusion_model_all_subjects"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Set DPI for faster plotting
    plot_dpi = 150  # Reduced from 300 for faster processing
    
    print("Step 1/4: Loading and organizing model paths...")
    
    # Get organized model paths
    _, model_df = organize_model_paths()
    
    # Get EEG-only model (Happyquoka) for reference
    eeg_only_model = model_df[model_df["model_type"].str.contains("Happyquoka")].iloc[0]["path"]
    
    # Get fusion models with different context lengths
    fusion_models = model_df[model_df["model_type"] == "TwoBranchContext"]
    
    # If there are no fusion models, exit
    if fusion_models.empty:
        print("No fusion models found")
        return
    
    print(f"Found {len(fusion_models)} fusion models with different context lengths")
    
    # Load EEG-only model data for all subjects
    print("Step 2/4: Loading EEG-only model data...")
    with open(eeg_only_model, 'rb') as file_handle:
        eeg_only_data = pickle.load(file_handle)
    
    # Define frequency bands for analysis
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    
    # Initialize lists to store data for averaging across subjects
    all_subjects_psd_data = {
        "ground_truth": [],
        "eeg_only": [],
        "fusion": {},
        "fusion_eeg_branch": {},
        "fusion_envelope_branch": {}
    }
    
    all_subjects_correlation_data = {
        "eeg_only": [],
        "fusion": {},
        "fusion_eeg_branch": {},
        "fusion_envelope_branch": {}
    }
    
    all_subjects_band_correlation_data = {
        "eeg_only": {band: [] for band in bands},
        "fusion": {band: {} for band in bands},
        "fusion_eeg_branch": {band: {} for band in bands},
        "fusion_envelope_branch": {band: {} for band in bands}
    }
    
    # Process each subject
    for subject_id in tqdm(all_subjects, desc="Processing subjects"):
        # Create subject-specific output directory
        subject_output_dir = os.path.join(base_output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        print(f"\nProcessing {subject_id}...")
        
        # Filter for current subject
        subject_eeg_only = filter_subject_data(eeg_only_data, subject_id)
        
        # Skip if no data is available for this subject
        if not subject_eeg_only["ground_truths"]:
            print(f"No data available for {subject_id}, skipping...")
            continue
        
        # Extract ground truths
        ground_truths = subject_eeg_only["ground_truths"]
        print(f"Found {len(ground_truths)} segments for {subject_id}")
        
        # Extract Happyquoka (EEG-only) model predictions
        eeg_only_predictions = [output["envelope"] for output in subject_eeg_only["outputs"]]
        
        # Compute PSD for ground truth
        psd_ground_truths, freq_values = compute_psd_welch(ground_truths, 64)
        
        # Compute PSD for Happyquoka model
        psd_eeg_only, _ = compute_psd_welch(eeg_only_predictions, 64)
        
        avg_psd_ground_truth = np.mean(psd_ground_truths, axis=0)
        avg_psd_eeg_only = np.mean(psd_eeg_only, axis=0)
        
        # Calculate correlation for Happyquoka model
        eeg_only_correlations = []
        for i in range(len(ground_truths)):
            gt = ground_truths[i].reshape(-1)
            pred = eeg_only_predictions[i].reshape(-1)
            corr = safe_correlation(gt, pred)
            eeg_only_correlations.append(corr)
        
        avg_eeg_only_correlation = np.mean(eeg_only_correlations)
        print(f"Happyquoka (EEG-only) model average correlation: {avg_eeg_only_correlation:.4f}")
        
        # Filter ground truth and EEG-only predictions into frequency bands
        ground_truth_bands = filter_signal_to_bands(ground_truths)
        eeg_only_bands = filter_signal_to_bands(eeg_only_predictions)
        
        # Calculate band-specific correlations for EEG-only model
        eeg_only_band_correlations = {}
        for band in bands:
            band_correlations = []
            for i in range(len(ground_truths)):
                gt_band = ground_truth_bands[band][i]
                pred_band = eeg_only_bands[band][i]
                corr = safe_correlation(gt_band, pred_band)
                band_correlations.append(corr)
            eeg_only_band_correlations[band] = band_correlations
        
        # Get context lengths from fusion models
        fusion_context_lengths = set()
        for _, row in fusion_models.iterrows():
            fusion_context_lengths.add(row["context_length"])
        
        context_lengths = sorted(list(fusion_context_lengths))
        print(f"Using {len(context_lengths)} context lengths for fusion models: {context_lengths}")
        
        # Load, filter, and compute PSD for each fusion model
        print(f"\nProcessing fusion models for {subject_id}...")
        fusion_results = {}
        
        # Process fusion models
        for _, row in fusion_models.iterrows():
            model_path = row["path"]
            context_length = row["context_length"]
            
            # Load model data
            with open(model_path, 'rb') as file_handle:
                model_data = pickle.load(file_handle)
            
            # Filter for current subject
            subject_data = filter_subject_data(model_data, subject_id)
            
            # Skip if no data for this subject
            if not subject_data["outputs"]:
                continue
                
            # Extract predictions for all branches
            predictions = [output["envelope"] for output in subject_data["outputs"]]
            eeg_branch_predictions = [output["eeg_branch"] for output in subject_data["outputs"]]
            envelope_branch_predictions = [output["envelope_branch"] for output in subject_data["outputs"]]
            
            # Compute PSD for all predictions
            psd_predictions, _ = compute_psd_welch(predictions, 64)
            psd_eeg_branch, _ = compute_psd_welch(eeg_branch_predictions, 64)
            psd_envelope_branch, _ = compute_psd_welch(envelope_branch_predictions, 64)
            
            avg_psd = np.mean(psd_predictions, axis=0)
            avg_psd_eeg_branch = np.mean(psd_eeg_branch, axis=0)
            avg_psd_envelope_branch = np.mean(psd_envelope_branch, axis=0)
            
            # Calculate correlations with ground truth for all branches
            correlations = []
            eeg_branch_correlations = []
            envelope_branch_correlations = []
            
            for i in range(len(ground_truths)):
                gt = ground_truths[i].reshape(-1)
                pred = predictions[i].reshape(-1)
                pred_eeg = eeg_branch_predictions[i].reshape(-1)
                pred_envelope = envelope_branch_predictions[i].reshape(-1)
                
                corr = safe_correlation(gt, pred)
                corr_eeg = safe_correlation(gt, pred_eeg)
                corr_envelope = safe_correlation(gt, pred_envelope)
                
                correlations.append(corr)
                eeg_branch_correlations.append(corr_eeg)
                envelope_branch_correlations.append(corr_envelope)
            
            avg_correlation = np.mean(correlations)
            avg_eeg_branch_correlation = np.mean(eeg_branch_correlations)
            avg_envelope_branch_correlation = np.mean(envelope_branch_correlations)
            
            # Filter predictions into frequency bands for all branches
            predictions_bands = filter_signal_to_bands(predictions)
            eeg_branch_bands = filter_signal_to_bands(eeg_branch_predictions)
            envelope_branch_bands = filter_signal_to_bands(envelope_branch_predictions)
            
            # Calculate band-specific correlations for all branches
            band_correlations = {}
            eeg_branch_band_correlations = {}
            envelope_branch_band_correlations = {}
            
            for band in bands:
                band_corrs = []
                eeg_branch_band_corrs = []
                envelope_branch_band_corrs = []
                
                for i in range(len(ground_truths)):
                    gt_band = ground_truth_bands[band][i]
                    pred_band = predictions_bands[band][i]
                    pred_eeg_band = eeg_branch_bands[band][i]
                    pred_envelope_band = envelope_branch_bands[band][i]
                    
                    corr = safe_correlation(gt_band, pred_band)
                    corr_eeg = safe_correlation(gt_band, pred_eeg_band)
                    corr_envelope = safe_correlation(gt_band, pred_envelope_band)
                    
                    band_corrs.append(corr)
                    eeg_branch_band_corrs.append(corr_eeg)
                    envelope_branch_band_corrs.append(corr_envelope)
                
                band_correlations[band] = band_corrs
                eeg_branch_band_correlations[band] = eeg_branch_band_corrs
                envelope_branch_band_correlations[band] = envelope_branch_band_corrs
            
            # Store results
            fusion_results[context_length] = {
                "predictions": predictions,
                "eeg_branch_predictions": eeg_branch_predictions,
                "envelope_branch_predictions": envelope_branch_predictions,
                "psd": psd_predictions,
                "psd_eeg_branch": psd_eeg_branch,
                "psd_envelope_branch": psd_envelope_branch,
                "avg_psd": avg_psd,
                "avg_psd_eeg_branch": avg_psd_eeg_branch,
                "avg_psd_envelope_branch": avg_psd_envelope_branch,
                "correlations": correlations,
                "eeg_branch_correlations": eeg_branch_correlations,
                "envelope_branch_correlations": envelope_branch_correlations,
                "avg_correlation": avg_correlation,
                "avg_eeg_branch_correlation": avg_eeg_branch_correlation,
                "avg_envelope_branch_correlation": avg_envelope_branch_correlation,
                "band_correlations": band_correlations,
                "eeg_branch_band_correlations": eeg_branch_band_correlations,
                "envelope_branch_band_correlations": envelope_branch_band_correlations
            }
        
        # Skip if no results for fusion models
        if not fusion_results:
            print(f"No results for {subject_id}, skipping plotting...")
            continue
            
        print(f"Generating plots for {subject_id}...")
        
        # Create color scheme for fusion models
        fusion_cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(context_lengths)))
        fusion_color = plt.cm.Oranges(0.7)  # Use a single darker orange for fusion line
        
        try:
            # Plot average PSDs
            plt.figure(figsize=(12, 8))
            plt.semilogy(freq_values, avg_psd_ground_truth, 'k-', linewidth=2, label="Ground Truth")
            plt.semilogy(freq_values, avg_psd_eeg_only, 'r-', linewidth=2, label="Happyquoka (EEG-only)")
            
            for i, context_length in enumerate(context_lengths):
                if context_length in fusion_results:
                    # Plot fusion model PSDs
                    plt.semilogy(freq_values, fusion_results[context_length]["avg_psd"], '-', 
                                color=fusion_cmap[i], label=f"Fusion {context_length:.2f}s")
                    plt.semilogy(freq_values, fusion_results[context_length]["avg_psd_eeg_branch"], '--', 
                                color=fusion_cmap[i], alpha=0.7, label=f"EEG Branch {context_length:.2f}s")
                    plt.semilogy(freq_values, fusion_results[context_length]["avg_psd_envelope_branch"], ':', 
                                color=fusion_cmap[i], alpha=0.7, label=f"Envelope Branch {context_length:.2f}s")
            
            plt.title(f"Average Power Spectral Density for Different Context Lengths ({subject_id})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density (log scale)")
            
            # Group similar model types together in the legend
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Re-order labels to group by model type
            try:
                eeg_only_idx = labels.index("Happyquoka (EEG-only)")
                ground_truth_idx = labels.index("Ground Truth")
                
                # First, put reference models
                new_order = [ground_truth_idx, eeg_only_idx]
                
                # Then add all fusion models and their branches
                fusion_indices = [i for i, label in enumerate(labels) if "Fusion" in label]
                eeg_branch_indices = [i for i, label in enumerate(labels) if "EEG Branch" in label]
                envelope_branch_indices = [i for i, label in enumerate(labels) if "Envelope Branch" in label]
                
                # Sort by context length
                fusion_indices.sort(key=lambda i: float(labels[i].split()[-1][:-1]))
                eeg_branch_indices.sort(key=lambda i: float(labels[i].split()[-1][:-1]))
                envelope_branch_indices.sort(key=lambda i: float(labels[i].split()[-1][:-1]))
                
                # Combine indices in desired order
                new_order.extend(fusion_indices)
                new_order.extend(eeg_branch_indices)
                new_order.extend(envelope_branch_indices)
                
                # Apply the new ordering
                handles = [handles[i] for i in new_order]
                labels = [labels[i] for i in new_order]
            except (ValueError, IndexError):
                # If there's an error in reordering, just use the original legend
                pass
            
            plt.legend(handles, labels, loc='best', fontsize=9)
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_fusion_psd.png"), dpi=plot_dpi)
            plt.close()
            
            # Plot correlation vs context length for all branches
            plt.figure(figsize=(10, 6))
            
            # Prepare data for all branches
            fusion_correlation_means = []
            fusion_correlation_stds = []
            eeg_branch_correlation_means = []
            eeg_branch_correlation_stds = []
            envelope_branch_correlation_means = []
            envelope_branch_correlation_stds = []
            
            for cl in context_lengths:
                if cl in fusion_results:
                    fusion_correlation_means.append(fusion_results[cl]["avg_correlation"])
                    fusion_correlation_stds.append(np.std(fusion_results[cl]["correlations"]))
                    eeg_branch_correlation_means.append(fusion_results[cl]["avg_eeg_branch_correlation"])
                    eeg_branch_correlation_stds.append(np.std(fusion_results[cl]["eeg_branch_correlations"]))
                    envelope_branch_correlation_means.append(fusion_results[cl]["avg_envelope_branch_correlation"])
                    envelope_branch_correlation_stds.append(np.std(fusion_results[cl]["envelope_branch_correlations"]))
                else:
                    fusion_correlation_means.append(np.nan)
                    fusion_correlation_stds.append(np.nan)
                    eeg_branch_correlation_means.append(np.nan)
                    eeg_branch_correlation_stds.append(np.nan)
                    envelope_branch_correlation_means.append(np.nan)
                    envelope_branch_correlation_stds.append(np.nan)
            
            # Plot correlation means with error bars for all branches
            plt.errorbar(context_lengths, fusion_correlation_means, yerr=fusion_correlation_stds, fmt='o-', 
                        capsize=5, markersize=8, linewidth=2, color=fusion_color, label="Fusion")
            plt.errorbar(context_lengths, eeg_branch_correlation_means, yerr=eeg_branch_correlation_stds, fmt='s--', 
                        capsize=5, markersize=8, linewidth=2, color=fusion_color, alpha=0.7, label="EEG Branch")
            plt.errorbar(context_lengths, envelope_branch_correlation_means, yerr=envelope_branch_correlation_stds, fmt='^:', 
                        capsize=5, markersize=8, linewidth=2, color=fusion_color, alpha=0.7, label="Envelope Branch")
            
            # Add horizontal line for EEG-only model
            plt.axhline(y=avg_eeg_only_correlation, color='r', linestyle='--', linewidth=2,
                       label=f"Happyquoka (EEG-only): {avg_eeg_only_correlation:.4f}")
            
            plt.title(f"Average Correlation with Ground Truth vs Context Length ({subject_id})")
            plt.xlabel("Context Length (seconds)")
            plt.ylabel("Pearson Correlation")
            
            # Add data point values as text labels
            for i, cl in enumerate(context_lengths):
                if not np.isnan(fusion_correlation_means[i]):
                    plt.text(cl, fusion_correlation_means[i] + 0.01, f"{fusion_correlation_means[i]:.3f}", 
                            ha='center', va='bottom', fontsize=7, color=fusion_color)
                    plt.text(cl, eeg_branch_correlation_means[i] + 0.01, f"{eeg_branch_correlation_means[i]:.3f}", 
                            ha='center', va='bottom', fontsize=7, color=fusion_color, alpha=0.7)
                    plt.text(cl, envelope_branch_correlation_means[i] + 0.01, f"{envelope_branch_correlation_means[i]:.3f}", 
                            ha='center', va='bottom', fontsize=7, color=fusion_color, alpha=0.7)
            
            plt.grid(True, ls="--")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_fusion_correlation.png"), dpi=plot_dpi)
            plt.close()
            
            # Bar plot of correlation performance by frequency band for all branches
            plt.figure(figsize=(10, 6))
            
            # Prepare bar positions and width
            bar_width = 0.15
            
            # Find best context length
            best_fusion_context = max(fusion_results.keys(), 
                                    key=lambda k: fusion_results[k]['avg_correlation'])
            
            # Bar positions
            bar_positions = np.arange(len(bands))
            
            # Prepare data for bars
            eeg_only_band_means = [np.mean(eeg_only_band_correlations[band]) for band in bands]
            eeg_only_band_stds = [np.std(eeg_only_band_correlations[band]) for band in bands]
            
            fusion_band_means = [np.mean(fusion_results[best_fusion_context]["band_correlations"][band]) 
                               for band in bands]
            fusion_band_stds = [np.std(fusion_results[best_fusion_context]["band_correlations"][band]) 
                              for band in bands]
            
            eeg_branch_band_means = [np.mean(fusion_results[best_fusion_context]["eeg_branch_band_correlations"][band]) 
                                   for band in bands]
            eeg_branch_band_stds = [np.std(fusion_results[best_fusion_context]["eeg_branch_band_correlations"][band]) 
                                  for band in bands]
            
            envelope_branch_band_means = [np.mean(fusion_results[best_fusion_context]["envelope_branch_band_correlations"][band]) 
                                        for band in bands]
            envelope_branch_band_stds = [np.std(fusion_results[best_fusion_context]["envelope_branch_band_correlations"][band]) 
                                       for band in bands]
            
            # Plot bars
            plt.bar(bar_positions - 1.5*bar_width, eeg_only_band_means, bar_width, 
                   yerr=eeg_only_band_stds, color='r', alpha=0.7, 
                   label=f"Happyquoka ({avg_eeg_only_correlation:.3f})")
            
            plt.bar(bar_positions - 0.5*bar_width, fusion_band_means, bar_width, 
                   yerr=fusion_band_stds, color=fusion_color, alpha=0.7,
                   label=f"Fusion {best_fusion_context:.2f}s ({fusion_results[best_fusion_context]['avg_correlation']:.3f})")
            
            plt.bar(bar_positions + 0.5*bar_width, eeg_branch_band_means, bar_width, 
                   yerr=eeg_branch_band_stds, color=fusion_color, alpha=0.5,
                   label=f"EEG Branch ({fusion_results[best_fusion_context]['avg_eeg_branch_correlation']:.3f})")
            
            plt.bar(bar_positions + 1.5*bar_width, envelope_branch_band_means, bar_width, 
                   yerr=envelope_branch_band_stds, color=fusion_color, alpha=0.3,
                   label=f"Envelope Branch ({fusion_results[best_fusion_context]['avg_envelope_branch_correlation']:.3f})")
            
            # Add correlation values as text on bars
            for i, band in enumerate(bands):
                # EEG-only
                plt.text(bar_positions[i] - 1.5*bar_width, eeg_only_band_means[i] + 0.01, 
                        f"{eeg_only_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # Fusion
                plt.text(bar_positions[i] - 0.5*bar_width, fusion_band_means[i] + 0.01, 
                        f"{fusion_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # EEG Branch
                plt.text(bar_positions[i] + 0.5*bar_width, eeg_branch_band_means[i] + 0.01, 
                        f"{eeg_branch_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # Envelope Branch
                plt.text(bar_positions[i] + 1.5*bar_width, envelope_branch_band_means[i] + 0.01, 
                        f"{envelope_branch_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
            
            plt.title(f"Correlation by Frequency Band ({subject_id})")
            plt.xlabel("Frequency Band")
            plt.ylabel("Pearson Correlation")
            plt.xticks(bar_positions, bands)
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_fusion_band_correlation_bars.png"), dpi=plot_dpi)
            plt.close()
            
            # Additional Bar Plot: Comparing all context lengths for fusion model
            plt.figure(figsize=(10, 6))
            
            # Number of contexts and bands
            n_contexts = len(fusion_results)
            n_bands = len(bands)
            
            # Bar positions
            bar_positions = np.arange(n_contexts) * (n_bands + 1)
            
            # Bar width
            bar_width = 0.8
            
            # Plot bars for each band
            for i, band in enumerate(bands):
                band_means = []
                band_stds = []
                
                for cl in sorted(fusion_results.keys()):
                    band_means.append(np.mean(fusion_results[cl]['band_correlations'][band]))
                    band_stds.append(np.std(fusion_results[cl]['band_correlations'][band]))
                
                # Different color for each band - using custom colormap
                band_color = plt.cm.viridis(i / len(bands))
                
                # Plot bars
                positions = bar_positions + i * bar_width
                plt.bar(positions, band_means, bar_width, yerr=band_stds, alpha=0.7, 
                       label=band, color=band_color)
                
                # Add correlation values as text
                for j, val in enumerate(band_means):
                    plt.text(positions[j], val + 0.01, f"{val:.3f}", ha='center', va='bottom', 
                            fontsize=6, color='black', rotation=90)
            
            # Set x-axis labels
            plt.xticks(bar_positions + (n_bands - 1) * bar_width / 2, 
                      [f"{cl:.2f}s" for cl in sorted(fusion_results.keys())])
            
            plt.title(f"Fusion Model: Correlation by Band and Context Length ({subject_id})")
            plt.xlabel("Context Length (seconds)")
            plt.ylabel("Pearson Correlation")
            plt.legend(title="Band", loc='upper right', fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_fusion_context_band_bars.png"), dpi=plot_dpi)
            plt.close()
            
            # Create table with correlation statistics
            correlation_data = {
                "Model": ["Happyquoka (EEG-only)"],
                "Mean Correlation": [f"{avg_eeg_only_correlation:.4f}"],
                "Std Dev": [f"{np.std(eeg_only_correlations):.4f}"],
                "Min": [f"{np.min(eeg_only_correlations):.4f}"],
                "Max": [f"{np.max(eeg_only_correlations):.4f}"]
            }
            
            # Create band-specific correlation data
            band_correlation_data = {band: {
                "Model": ["Happyquoka (EEG-only)"],
                "Mean Correlation": [f"{np.mean(eeg_only_band_correlations[band]):.4f}"],
                "Std Dev": [f"{np.std(eeg_only_band_correlations[band]):.4f}"],
                "Min": [f"{np.min(eeg_only_band_correlations[band]):.4f}"],
                "Max": [f"{np.max(eeg_only_band_correlations[band]):.4f}"]
            } for band in bands}
            
            # Add fusion models to correlation data
            for cl in context_lengths:
                if cl in fusion_results:
                    correlation_data["Model"].append(f"Fusion Context {cl:.2f}s")
                    correlation_data["Mean Correlation"].append(f"{fusion_results[cl]['avg_correlation']:.4f}")
                    correlation_data["Std Dev"].append(f"{np.std(fusion_results[cl]['correlations']):.4f}")
                    correlation_data["Min"].append(f"{np.min(fusion_results[cl]['correlations']):.4f}")
                    correlation_data["Max"].append(f"{np.max(fusion_results[cl]['correlations']):.4f}")
                    
                    # Add band-specific data
                    for band in bands:
                        band_correlations = fusion_results[cl]['band_correlations'][band]
                        band_correlation_data[band]["Model"].append(f"Fusion Context {cl:.2f}s")
                        band_correlation_data[band]["Mean Correlation"].append(f"{np.mean(band_correlations):.4f}")
                        band_correlation_data[band]["Std Dev"].append(f"{np.std(band_correlations):.4f}")
                        band_correlation_data[band]["Min"].append(f"{np.min(band_correlations):.4f}")
                        band_correlation_data[band]["Max"].append(f"{np.max(band_correlations):.4f}")
            
            # Save correlation statistics to a text file
            with open(os.path.join(subject_output_dir, f"{subject_id}_fusion_correlation_statistics.txt"), 'w') as f:
                f.write(f"Correlation Statistics for {subject_id}:\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Model':<20} {'Mean Correlation':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}\n")
                f.write("-" * 90 + "\n")
                
                for i in range(len(correlation_data["Model"])):
                    f.write(f"{correlation_data['Model'][i]:<20} "
                            f"{correlation_data['Mean Correlation'][i]:<20} "
                            f"{correlation_data['Std Dev'][i]:<15} "
                            f"{correlation_data['Min'][i]:<15} "
                            f"{correlation_data['Max'][i]:<15}\n")
                
                # Add band-specific statistics
                for band in bands:
                    f.write("\n\n")
                    f.write(f"{band.upper()} Band Correlation Statistics:\n")
                    f.write("-" * 90 + "\n")
                    f.write(f"{'Model':<20} {'Mean Correlation':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}\n")
                    f.write("-" * 90 + "\n")
                    
                    for i in range(len(band_correlation_data[band]["Model"])):
                        f.write(f"{band_correlation_data[band]['Model'][i]:<20} "
                                f"{band_correlation_data[band]['Mean Correlation'][i]:<20} "
                                f"{band_correlation_data[band]['Std Dev'][i]:<15} "
                                f"{band_correlation_data[band]['Min'][i]:<15} "
                                f"{band_correlation_data[band]['Max'][i]:<15}\n")
        
        except Exception as e:
            print(f"Error during plotting for {subject_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Store data for averaging across subjects
        all_subjects_psd_data["ground_truth"].append(psd_ground_truths)
        all_subjects_psd_data["eeg_only"].append(psd_eeg_only)
        for context_length in context_lengths:
            if context_length in fusion_results:
                if context_length not in all_subjects_psd_data["fusion"]:
                    all_subjects_psd_data["fusion"][context_length] = []
                all_subjects_psd_data["fusion"][context_length].extend(fusion_results[context_length]["psd"])
        
        all_subjects_correlation_data["eeg_only"].extend(eeg_only_correlations)
        for context_length in context_lengths:
            if context_length in fusion_results:
                if context_length not in all_subjects_correlation_data["fusion"]:
                    all_subjects_correlation_data["fusion"][context_length] = []
                all_subjects_correlation_data["fusion"][context_length].extend(fusion_results[context_length]["correlations"])
        
        for band in bands:
            all_subjects_band_correlation_data["eeg_only"][band].extend(eeg_only_band_correlations[band])
            for context_length in context_lengths:
                if context_length in fusion_results:
                    if context_length not in all_subjects_band_correlation_data["fusion"][band]:
                        all_subjects_band_correlation_data["fusion"][band][context_length] = []
                    all_subjects_band_correlation_data["fusion"][band][context_length].extend(
                        fusion_results[context_length]["band_correlations"][band])
    
    # After processing all subjects, create average plots
    print("\nGenerating average plots across all subjects...")
    
    # Create directory for average plots
    avg_plots_dir = os.path.join(base_output_dir, "average_plots")
    os.makedirs(avg_plots_dir, exist_ok=True)
    
    try:
        # Average PSD Plot
        plt.figure(figsize=(12, 8))
        
        # Calculate averages
        avg_psd_ground_truth = np.mean([psd for psd in all_subjects_psd_data["ground_truth"]], axis=0)
        avg_psd_eeg_only = np.mean([psd for psd in all_subjects_psd_data["eeg_only"]], axis=0)
        
        # Plot averages
        plt.semilogy(freq_values, avg_psd_ground_truth, 'k-', linewidth=2, label="Ground Truth")
        plt.semilogy(freq_values, avg_psd_eeg_only, 'r-', linewidth=2, label="Happyquoka (EEG-only)")
        
        # Plot fusion models
        for context_length in context_lengths:
            if context_length in all_subjects_psd_data["fusion"]:
                avg_psd = np.mean([psd for psd in all_subjects_psd_data["fusion"][context_length]], axis=0)
                plt.semilogy(freq_values, avg_psd, '-', 
                            color=fusion_cmap[context_lengths.index(context_length)], 
                            label=f"Fusion {context_length:.2f}s")
        
        plt.title("Average Power Spectral Density Across All Subjects")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (log scale)")
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_fusion_psd.png"), dpi=plot_dpi)
        plt.close()
        
        # Average Correlation Plot
        plt.figure(figsize=(10, 6))
        
        # Calculate averages for each model type
        avg_eeg_only_corr = np.mean(all_subjects_correlation_data["eeg_only"])
        
        # Plot correlation means with error bars
        for context_length in context_lengths:
            if context_length in all_subjects_correlation_data["fusion"]:
                corrs = all_subjects_correlation_data["fusion"][context_length]
                plt.errorbar(context_length, np.mean(corrs), yerr=np.std(corrs), fmt='o-', 
                            capsize=5, markersize=8, linewidth=2, color=fusion_color, 
                            label="Fusion" if context_length == context_lengths[0] else "")
        
        # Add horizontal line for EEG-only model
        plt.axhline(y=avg_eeg_only_corr, color='r', linestyle='--', linewidth=2,
                   label=f"Happyquoka (EEG-only): {avg_eeg_only_corr:.4f}")
        
        plt.title("Average Correlation with Ground Truth vs Context Length (All Subjects)")
        plt.xlabel("Context Length (seconds)")
        plt.ylabel("Pearson Correlation")
        plt.grid(True, ls="--")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_fusion_correlation.png"), dpi=plot_dpi)
        plt.close()
        
        # Average Band Correlation Bar Plot
        plt.figure(figsize=(10, 6))
        
        # Calculate averages for each band and model type
        bar_positions = np.arange(len(bands))
        bar_width = 0.15
        
        # EEG-only band means
        eeg_only_band_means = [np.mean(all_subjects_band_correlation_data["eeg_only"][band]) 
                             for band in bands]
        eeg_only_band_stds = [np.std(all_subjects_band_correlation_data["eeg_only"][band]) 
                            for band in bands]
        
        # Plot EEG-only bars
        plt.bar(bar_positions - 0.5*bar_width, eeg_only_band_means, bar_width, 
               yerr=eeg_only_band_stds, color='r', alpha=0.7, 
               label=f"Happyquoka ({avg_eeg_only_corr:.3f})")
        
        # Find best context length for fusion model
        best_fusion_context = max(all_subjects_correlation_data["fusion"].keys(), 
                                 key=lambda k: np.mean(all_subjects_correlation_data["fusion"][k]))
        
        # Calculate and plot fusion band means
        fusion_band_means = []
        fusion_band_stds = []
        for band in bands:
            corrs = all_subjects_band_correlation_data["fusion"][band][best_fusion_context]
            fusion_band_means.append(np.mean(corrs))
            fusion_band_stds.append(np.std(corrs))
        
        plt.bar(bar_positions + 0.5*bar_width, fusion_band_means, bar_width, 
               yerr=fusion_band_stds, color=fusion_color, alpha=0.7,
               label=f"Fusion {best_fusion_context:.2f}s")
        
        # Add correlation values as text on bars
        for i, band in enumerate(bands):
            # EEG-only
            plt.text(bar_positions[i] - 0.5*bar_width, eeg_only_band_means[i] + 0.01, 
                    f"{eeg_only_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
            
            # Fusion
            plt.text(bar_positions[i] + 0.5*bar_width, fusion_band_means[i] + 0.01, 
                    f"{fusion_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
        
        plt.title("Average Correlation by Frequency Band (All Subjects)")
        plt.xlabel("Frequency Band")
        plt.ylabel("Pearson Correlation")
        plt.xticks(bar_positions, bands)
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_fusion_band_correlation_bars.png"), dpi=plot_dpi)
        plt.close()
        
        # Save average statistics to a text file
        with open(os.path.join(avg_plots_dir, "average_fusion_correlation_statistics.txt"), 'w') as f:
            f.write("Average Correlation Statistics Across All Subjects:\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Model':<20} {'Mean Correlation':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}\n")
            f.write("-" * 90 + "\n")
            
            # Write EEG-only statistics
            f.write(f"{'Happyquoka (EEG-only)':<20} "
                    f"{avg_eeg_only_corr:.4f:<20} "
                    f"{np.std(all_subjects_correlation_data['eeg_only']):.4f:<15} "
                    f"{np.min(all_subjects_correlation_data['eeg_only']):.4f:<15} "
                    f"{np.max(all_subjects_correlation_data['eeg_only']):.4f:<15}\n")
            
            # Write fusion model statistics
            for context_length in context_lengths:
                if context_length in all_subjects_correlation_data["fusion"]:
                    corrs = all_subjects_correlation_data["fusion"][context_length]
                    f.write(f"{f'Fusion {context_length:.2f}s':<20} "
                            f"{np.mean(corrs):.4f:<20} "
                            f"{np.std(corrs):.4f:<15} "
                            f"{np.min(corrs):.4f:<15} "
                            f"{np.max(corrs):.4f:<15}\n")
            
            # Write band-specific statistics
            for band in bands:
                f.write("\n\n")
                f.write(f"{band.upper()} Band Average Correlation Statistics:\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Model':<20} {'Mean Correlation':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}\n")
                f.write("-" * 90 + "\n")
                
                # EEG-only band statistics
                corrs = all_subjects_band_correlation_data["eeg_only"][band]
                f.write(f"{'Happyquoka (EEG-only)':<20} "
                        f"{np.mean(corrs):.4f:<20} "
                        f"{np.std(corrs):.4f:<15} "
                        f"{np.min(corrs):.4f:<15} "
                        f"{np.max(corrs):.4f:<15}\n")
                
                # Fusion band statistics
                for context_length in context_lengths:
                    if context_length in all_subjects_band_correlation_data["fusion"][band]:
                        corrs = all_subjects_band_correlation_data["fusion"][band][context_length]
                        f.write(f"{f'Fusion {context_length:.2f}s':<20} "
                                f"{np.mean(corrs):.4f:<20} "
                                f"{np.std(corrs):.4f:<15} "
                                f"{np.min(corrs):.4f:<15} "
                                f"{np.max(corrs):.4f:<15}\n")
    
    except Exception as e:
        print(f"Error during average plotting: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nProcessing completed. Results saved to:", base_output_dir)

if __name__ == "__main__":
    analyze_fusion_model_for_all_subjects() 