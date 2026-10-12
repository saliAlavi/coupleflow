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
        freqs, Pxx = signal.welch(x.reshape(-1), sampling_rate, nperseg=64, noverlap=32)
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

def compare_envelope_models_for_all_subjects():
    """Compare envelope context models with different context lengths for all subjects"""
    
    # List of subjects to process (sub-001 to sub-085)
    all_subjects = [f"sub-{i:03d}" for i in range(1, 86)]
    
    # Create output directory structure
    base_output_dir = "plots_all_subjects"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Set DPI for faster plotting
    plot_dpi = 150  # Reduced from 300 for faster processing
    
    print("Step 1/5: Loading and organizing model paths...")
    
    # Get organized model paths
    _, model_df = organize_model_paths()
    
    # Get EEG-only model (Happyquoka) for reference
    eeg_only_model = model_df[model_df["model_type"].str.contains("Happyquoka")].iloc[0]["path"]
    
    # Get envelope models with different context lengths
    envelope_models = model_df[model_df["model_type"] == "TwoBranchFrozenEegContext_EnvelopeOnly"]
    
    # Get fusion models with different context lengths
    fusion_models = model_df[model_df["model_type"] == "TwoBranchContext"]
    
    # Get frozen EEG models with different context lengths
    frozen_eeg_models = model_df[model_df["model_type"] == "TwoBranchFrozenEegContext_Dynamic"]
    
    # If there are no envelope models, exit
    if envelope_models.empty:
        print("No envelope context models found")
        return
    
    print(f"Found {len(envelope_models)} envelope context models with different context lengths")
    print(f"Found {len(fusion_models)} fusion context models with different context lengths")
    print(f"Found {len(frozen_eeg_models)} frozen EEG context models with different context lengths")
    
    # Load EEG-only model data for all subjects
    print("Step 2/5: Loading EEG-only model data...")
    with open(eeg_only_model, 'rb') as file_handle:
        eeg_only_data = pickle.load(file_handle)
    
    # Define frequency bands for analysis
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    
    # Initialize lists to store data for averaging across subjects
    all_subjects_psd_data = {
        "ground_truth": [],
        "eeg_only": [],
        "envelope": {},
        "fusion": {},
        "frozen_eeg": {}
    }
    
    all_subjects_correlation_data = {
        "eeg_only": [],
        "envelope": {},
        "fusion": {},
        "frozen_eeg": {}
    }
    
    all_subjects_band_correlation_data = {
        "eeg_only": {band: [] for band in ["delta", "theta", "alpha", "beta", "gamma"]},
        "envelope": {band: {} for band in ["delta", "theta", "alpha", "beta", "gamma"]},
        "fusion": {band: {} for band in ["delta", "theta", "alpha", "beta", "gamma"]},
        "frozen_eeg": {band: {} for band in ["delta", "theta", "alpha", "beta", "gamma"]}
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
        
        # Process, load and gather all context lengths across models
        # Get context lengths from envelope models
        envelope_context_lengths = set()
        for _, row in envelope_models.iterrows():
            envelope_context_lengths.add(row["context_length"])
        
        # Get context lengths from fusion models
        fusion_context_lengths = set()
        for _, row in fusion_models.iterrows():
            fusion_context_lengths.add(row["context_length"])
        
        # Get context lengths from frozen EEG models
        frozen_eeg_context_lengths = set()
        for _, row in frozen_eeg_models.iterrows():
            frozen_eeg_context_lengths.add(row["context_length"])
        
        # Find common context lengths across all model types to avoid KeyErrors
        context_lengths = sorted(list(envelope_context_lengths.intersection(fusion_context_lengths)))
        print(f"Using {len(context_lengths)} common context lengths across envelope and fusion models: {context_lengths}")
        
        # Load, filter, and compute PSD for each envelope and fusion model
        print(f"\nProcessing envelope and fusion models for {subject_id}...")
        envelope_results = {}
        fusion_results = {}
        frozen_eeg_results = {}
        
        # Process envelope models
        for _, row in envelope_models.iterrows():
            model_path = row["path"]
            context_length = row["context_length"]
            
            # Skip if not in common context lengths
            if context_length not in context_lengths:
                continue
                
            # Load model data
            with open(model_path, 'rb') as file_handle:
                model_data = pickle.load(file_handle)
            
            # Filter for current subject
            subject_data = filter_subject_data(model_data, subject_id)
            
            # Skip if no data for this subject
            if not subject_data["outputs"]:
                continue
                
            # Extract predictions
            predictions = [output["envelope"] for output in subject_data["outputs"]]
            
            # Compute PSD
            psd_predictions, _ = compute_psd_welch(predictions, 64)
            avg_psd = np.mean(psd_predictions, axis=0)
            
            # Calculate correlations with ground truth
            correlations = []
            for i in range(len(ground_truths)):
                gt = ground_truths[i].reshape(-1)
                pred = predictions[i].reshape(-1)
                corr = safe_correlation(gt, pred)
                correlations.append(corr)
            
            avg_correlation = np.mean(correlations)
            
            # Filter predictions into frequency bands
            predictions_bands = filter_signal_to_bands(predictions)
            
            # Calculate band-specific correlations
            band_correlations = {}
            for band in bands:
                band_corrs = []
                for i in range(len(ground_truths)):
                    gt_band = ground_truth_bands[band][i]
                    pred_band = predictions_bands[band][i]
                    corr = safe_correlation(gt_band, pred_band)
                    band_corrs.append(corr)
                band_correlations[band] = band_corrs
            
            # Store results
            envelope_results[context_length] = {
                "predictions": predictions,
                "psd": psd_predictions,
                "avg_psd": avg_psd,
                "correlations": correlations,
                "avg_correlation": avg_correlation,
                "band_correlations": band_correlations
            }
        
        # Process fusion models
        for _, row in fusion_models.iterrows():
            model_path = row["path"]
            context_length = row["context_length"]
            
            # Skip if not in common context lengths
            if context_length not in context_lengths:
                continue
                
            # Load model data
            with open(model_path, 'rb') as file_handle:
                model_data = pickle.load(file_handle)
            
            # Filter for current subject
            subject_data = filter_subject_data(model_data, subject_id)
            
            # Skip if no data for this subject
            if not subject_data["outputs"]:
                continue
            
            # Extract predictions
            predictions = [output["envelope"] for output in subject_data["outputs"]]
            
            # Compute PSD
            psd_predictions, _ = compute_psd_welch(predictions, 64)
            avg_psd = np.mean(psd_predictions, axis=0)
            
            # Calculate correlations with ground truth
            correlations = []
            for i in range(len(ground_truths)):
                gt = ground_truths[i].reshape(-1)
                pred = predictions[i].reshape(-1)
                corr = safe_correlation(gt, pred)
                correlations.append(corr)
            
            avg_correlation = np.mean(correlations)
            
            # Filter predictions into frequency bands
            predictions_bands = filter_signal_to_bands(predictions)
            
            # Calculate band-specific correlations
            band_correlations = {}
            for band in bands:
                band_corrs = []
                for i in range(len(ground_truths)):
                    gt_band = ground_truth_bands[band][i]
                    pred_band = predictions_bands[band][i]
                    corr = safe_correlation(gt_band, pred_band)
                    band_corrs.append(corr)
                band_correlations[band] = band_corrs
            
            # Store results
            fusion_results[context_length] = {
                "predictions": predictions,
                "psd": psd_predictions,
                "avg_psd": avg_psd,
                "correlations": correlations,
                "avg_correlation": avg_correlation,
                "band_correlations": band_correlations
            }
        
        # Process frozen EEG models
        for _, row in frozen_eeg_models.iterrows():
            model_path = row["path"]
            context_length = row["context_length"]
            
            # Skip if not in common context lengths
            if context_length not in context_lengths:
                continue
                
            # Load model data
            with open(model_path, 'rb') as file_handle:
                model_data = pickle.load(file_handle)
            
            # Filter for current subject
            subject_data = filter_subject_data(model_data, subject_id)
            
            # Skip if no data for this subject
            if not subject_data["outputs"]:
                continue
            
            # Extract predictions
            predictions = [output["envelope"] for output in subject_data["outputs"]]
            
            # Compute PSD
            psd_predictions, _ = compute_psd_welch(predictions, 64)
            avg_psd = np.mean(psd_predictions, axis=0)
            
            # Calculate correlations with ground truth
            correlations = []
            for i in range(len(ground_truths)):
                gt = ground_truths[i].reshape(-1)
                pred = predictions[i].reshape(-1)
                corr = safe_correlation(gt, pred)
                correlations.append(corr)
            
            avg_correlation = np.mean(correlations)
            
            # Filter predictions into frequency bands
            predictions_bands = filter_signal_to_bands(predictions)
            
            # Calculate band-specific correlations
            band_correlations = {}
            for band in bands:
                band_corrs = []
                for i in range(len(ground_truths)):
                    gt_band = ground_truth_bands[band][i]
                    pred_band = predictions_bands[band][i]
                    corr = safe_correlation(gt_band, pred_band)
                    band_corrs.append(corr)
                band_correlations[band] = band_corrs
            
            # Store results
            frozen_eeg_results[context_length] = {
                "predictions": predictions,
                "psd": psd_predictions,
                "avg_psd": avg_psd,
                "correlations": correlations,
                "avg_correlation": avg_correlation,
                "band_correlations": band_correlations
            }
        
        # Skip if no results for any of the models
        if not envelope_results or not fusion_results:
            print(f"No results for {subject_id}, skipping plotting...")
            continue
            
        print(f"Generating plots for {subject_id}...")
        
        # Create better color schemes for three model types
        envelope_cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(context_lengths)))
        fusion_cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(context_lengths)))
        frozen_eeg_cmap = plt.cm.Purples(np.linspace(0.4, 0.9, len(context_lengths)))  # Purple for frozen EEG
        
        try:
            # Plot average PSDs
            plt.figure(figsize=(12, 8))
            plt.semilogy(freq_values, avg_psd_ground_truth, 'k-', linewidth=2, label="Ground Truth")
            plt.semilogy(freq_values, avg_psd_eeg_only, 'r-', linewidth=2, label="Happyquoka (EEG-only)")
            
            for i, context_length in enumerate(context_lengths):
                if context_length in envelope_results:
                    plt.semilogy(freq_values, envelope_results[context_length]["avg_psd"], '-', 
                                color=envelope_cmap[i], label=f"Envelope {context_length:.2f}s")
                if context_length in fusion_results:
                    plt.semilogy(freq_values, fusion_results[context_length]["avg_psd"], '-', 
                                color=fusion_cmap[i], label=f"Fusion {context_length:.2f}s")
                if context_length in frozen_eeg_results:
                    plt.semilogy(freq_values, frozen_eeg_results[context_length]["avg_psd"], '-', 
                                color=frozen_eeg_cmap[i], label=f"Frozen EEG {context_length:.2f}s")
            
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
                
                # Then add all envelope models
                envelope_indices = [i for i, label in enumerate(labels) if "Envelope" in label]
                new_order.extend(sorted(envelope_indices, key=lambda i: float(labels[i].split()[-1][:-1])))
                
                # Then add all fusion models
                fusion_indices = [i for i, label in enumerate(labels) if "Fusion" in label]
                new_order.extend(sorted(fusion_indices, key=lambda i: float(labels[i].split()[-1][:-1])))
                
                # Finally add all frozen EEG models
                frozen_indices = [i for i, label in enumerate(labels) if "Frozen EEG" in label]
                new_order.extend(sorted(frozen_indices, key=lambda i: float(labels[i].split()[-1][:-1])))
                
                # Apply the new ordering
                handles = [handles[i] for i in new_order]
                labels = [labels[i] for i in new_order]
            except (ValueError, IndexError):
                # If there's an error in reordering, just use the original legend
                pass
            
            plt.legend(handles, labels, loc='best', fontsize=9)
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_context_comparison_psd.png"), dpi=plot_dpi)
            plt.close()
            
            # Plot correlation vs context length
            plt.figure(figsize=(10, 6))
            
            # Create better color schemes for three model types - using darker colors for line plots
            envelope_color = plt.cm.Blues(0.7)  # Use a single darker blue for envelope line
            fusion_color = plt.cm.Oranges(0.7)  # Use a single darker orange for fusion line
            frozen_eeg_color = plt.cm.Purples(0.7)  # Use a single darker purple for frozen EEG line
            
            # Prepare data for each model type
            envelope_correlation_means = []
            envelope_correlation_stds = []
            fusion_correlation_means = []
            fusion_correlation_stds = []
            frozen_eeg_correlation_means = []
            frozen_eeg_correlation_stds = []
            
            for cl in context_lengths:
                # Envelope models
                if cl in envelope_results:
                    envelope_correlation_means.append(envelope_results[cl]["avg_correlation"])
                    envelope_correlation_stds.append(np.std(envelope_results[cl]["correlations"]))
                else:
                    envelope_correlation_means.append(np.nan)
                    envelope_correlation_stds.append(np.nan)
                
                # Fusion models
                if cl in fusion_results:
                    fusion_correlation_means.append(fusion_results[cl]["avg_correlation"])
                    fusion_correlation_stds.append(np.std(fusion_results[cl]["correlations"]))
                else:
                    fusion_correlation_means.append(np.nan)
                    fusion_correlation_stds.append(np.nan)
                
                # Frozen EEG models
                if cl in frozen_eeg_results:
                    frozen_eeg_correlation_means.append(frozen_eeg_results[cl]["avg_correlation"])
                    frozen_eeg_correlation_stds.append(np.std(frozen_eeg_results[cl]["correlations"]))
                else:
                    frozen_eeg_correlation_means.append(np.nan)
                    frozen_eeg_correlation_stds.append(np.nan)
            
            # Plot correlation means with error bars
            plt.errorbar(context_lengths, envelope_correlation_means, yerr=envelope_correlation_stds, fmt='o-', 
                        capsize=5, markersize=8, linewidth=2, color=envelope_color, label="Envelope")
            plt.errorbar(context_lengths, fusion_correlation_means, yerr=fusion_correlation_stds, fmt='o-', 
                        capsize=5, markersize=8, linewidth=2, color=fusion_color, label="Fusion")
            plt.errorbar(context_lengths, frozen_eeg_correlation_means, yerr=frozen_eeg_correlation_stds, fmt='o-', 
                        capsize=5, markersize=8, linewidth=2, color=frozen_eeg_color, label="Frozen EEG")
            
            # Add horizontal line for EEG-only (Happyquoka) model
            plt.axhline(y=avg_eeg_only_correlation, color='r', linestyle='--', linewidth=2,
                       label=f"Happyquoka (EEG-only): {avg_eeg_only_correlation:.4f}")
            
            plt.title(f"Average Correlation with Ground Truth vs Context Length ({subject_id})")
            plt.xlabel("Context Length (seconds)")
            plt.ylabel("Pearson Correlation")
            
            # Add data point values as text labels
            for i, cl in enumerate(context_lengths):
                # Envelope model labels (slightly offset to avoid overlap)
                if not np.isnan(envelope_correlation_means[i]):
                    plt.text(cl-0.02, envelope_correlation_means[i] - 0.015, f"{envelope_correlation_means[i]:.3f}", 
                            ha='right', va='top', fontsize=7, color=envelope_color)
                
                # Fusion model labels
                if not np.isnan(fusion_correlation_means[i]):
                    plt.text(cl, fusion_correlation_means[i] + 0.01, f"{fusion_correlation_means[i]:.3f}", 
                            ha='center', va='bottom', fontsize=7, color=fusion_color)
                
                # Frozen EEG model labels (slightly offset to avoid overlap)
                if not np.isnan(frozen_eeg_correlation_means[i]):
                    plt.text(cl+0.02, frozen_eeg_correlation_means[i] - 0.015, f"{frozen_eeg_correlation_means[i]:.3f}", 
                            ha='left', va='top', fontsize=7, color=frozen_eeg_color)
            
            plt.grid(True, ls="--")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_context_correlation.png"), dpi=plot_dpi)
            plt.close()
            
            # New: Bar plot of correlation performance by model type
            plt.figure(figsize=(10, 6))
            
            # Prepare bar positions and width
            bar_width = 0.15
            
            # Choose one context length for comparison (using the best performing one)
            # Find best context length for each model type
            best_envelope_context = max(envelope_results.keys(), 
                                      key=lambda k: envelope_results[k]['avg_correlation'])
            best_fusion_context = max(fusion_results.keys(), 
                                     key=lambda k: fusion_results[k]['avg_correlation'])
            
            # Best frozen EEG context (if available)
            best_frozen_eeg_context = None
            if frozen_eeg_results:
                best_frozen_eeg_context = max(frozen_eeg_results.keys(), 
                                           key=lambda k: frozen_eeg_results[k]['avg_correlation'])
            
            # Bar positions
            bar_positions = np.arange(len(bands))
            
            # Prepare data for bars
            eeg_only_band_means = [np.mean([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) 
                                  for band in bands]
            eeg_only_band_stds = [np.std([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) 
                                 for band in bands]
            
            envelope_band_means = [np.mean([corr for corr in envelope_results[best_envelope_context]["band_correlations"][band] if not np.isnan(corr)]) 
                                  for band in bands]
            envelope_band_stds = [np.std([corr for corr in envelope_results[best_envelope_context]["band_correlations"][band] if not np.isnan(corr)]) 
                                 for band in bands]
            
            fusion_band_means = [np.mean([corr for corr in fusion_results[best_fusion_context]["band_correlations"][band] if not np.isnan(corr)]) 
                               for band in bands]
            fusion_band_stds = [np.std([corr for corr in fusion_results[best_fusion_context]["band_correlations"][band] if not np.isnan(corr)]) 
                              for band in bands]
            
            # Plot bars
            plt.bar(bar_positions - 1.5*bar_width, eeg_only_band_means, bar_width, 
                   yerr=eeg_only_band_stds, color='r', alpha=0.7, 
                   label=f"Happyquoka ({avg_eeg_only_correlation:.3f})")
            
            plt.bar(bar_positions - 0.5*bar_width, envelope_band_means, bar_width, 
                   yerr=envelope_band_stds, color=envelope_color, alpha=0.7,
                   label=f"Envelope {best_envelope_context:.2f}s ({envelope_results[best_envelope_context]['avg_correlation']:.3f})")
            
            plt.bar(bar_positions + 0.5*bar_width, fusion_band_means, bar_width, 
                   yerr=fusion_band_stds, color=fusion_color, alpha=0.7,
                   label=f"Fusion {best_fusion_context:.2f}s ({fusion_results[best_fusion_context]['avg_correlation']:.3f})")
            
            # Add frozen EEG bars if available
            if best_frozen_eeg_context:
                frozen_eeg_band_means = [np.mean([corr for corr in frozen_eeg_results[best_frozen_eeg_context]["band_correlations"][band] if not np.isnan(corr)]) 
                                       for band in bands]
                frozen_eeg_band_stds = [np.std([corr for corr in frozen_eeg_results[best_frozen_eeg_context]["band_correlations"][band] if not np.isnan(corr)]) 
                                      for band in bands]
                
                plt.bar(bar_positions + 1.5*bar_width, frozen_eeg_band_means, bar_width, 
                       yerr=frozen_eeg_band_stds, color=frozen_eeg_color, alpha=0.7,
                       label=f"Frozen EEG {best_frozen_eeg_context:.2f}s ({frozen_eeg_results[best_frozen_eeg_context]['avg_correlation']:.3f})")
            
            # Add correlation values as text on bars
            for i, band in enumerate(bands):
                # EEG-only
                plt.text(bar_positions[i] - 1.5*bar_width, eeg_only_band_means[i] + 0.01, 
                        f"{eeg_only_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # Envelope
                plt.text(bar_positions[i] - 0.5*bar_width, envelope_band_means[i] + 0.01, 
                        f"{envelope_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # Fusion
                plt.text(bar_positions[i] + 0.5*bar_width, fusion_band_means[i] + 0.01, 
                        f"{fusion_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
                
                # Frozen EEG
                if best_frozen_eeg_context:
                    plt.text(bar_positions[i] + 1.5*bar_width, frozen_eeg_band_means[i] + 0.01, 
                            f"{frozen_eeg_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
            
            plt.title(f"Correlation by Frequency Band ({subject_id})")
            plt.xlabel("Frequency Band")
            plt.ylabel("Pearson Correlation")
            plt.xticks(bar_positions, bands)
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_band_correlation_bars.png"), dpi=plot_dpi)
            plt.close()

            # Additional Bar Plot: Comparing all context lengths for best model type
            # (choose the best model type based on average correlation)
            avg_envelope_corr = np.mean([envelope_results[cl]["avg_correlation"] for cl in envelope_results])
            avg_fusion_corr = np.mean([fusion_results[cl]["avg_correlation"] for cl in fusion_results])
            
            best_model_type = "fusion" if avg_fusion_corr > avg_envelope_corr else "envelope"
            
            if best_model_type == "fusion":
                best_model_results = fusion_results
                best_model_color = fusion_color
                best_model_name = "Fusion"
            else:
                best_model_results = envelope_results
                best_model_color = envelope_color
                best_model_name = "Envelope"
            
            plt.figure(figsize=(10, 6))
            
            # Number of contexts and bands
            n_contexts = len(best_model_results)
            n_bands = len(bands)
            
            # Bar positions
            bar_positions = np.arange(n_contexts) * (n_bands + 1)
            
            # Bar width
            bar_width = 0.8
            
            # Plot bars for each band
            for i, band in enumerate(bands):
                band_means = []
                band_stds = []
                
                for cl in sorted(best_model_results.keys()):
                    band_means.append(np.mean([corr for corr in best_model_results[cl]['band_correlations'][band] if not np.isnan(corr)]))
                    band_stds.append(np.std([corr for corr in best_model_results[cl]['band_correlations'][band] if not np.isnan(corr)]))
                
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
                      [f"{cl:.2f}s" for cl in sorted(best_model_results.keys())])
            
            plt.title(f"{best_model_name} Models: Correlation by Band and Context Length ({subject_id})")
            plt.xlabel("Context Length (seconds)")
            plt.ylabel("Pearson Correlation")
            plt.legend(title="Band", loc='upper right', fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, f"{subject_id}_{best_model_name.lower()}_context_band_bars.png"), dpi=plot_dpi)
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
                "Mean Correlation": [f"{np.mean([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) if eeg_only_band_correlations[band] else np.nan:.4f}"],
                "Std Dev": [f"{np.std([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) if eeg_only_band_correlations[band] else np.nan:.4f}"],
                "Min": [f"{np.min([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) if eeg_only_band_correlations[band] else np.nan:.4f}"],
                "Max": [f"{np.max([corr for corr in eeg_only_band_correlations[band] if not np.isnan(corr)]) if eeg_only_band_correlations[band] else np.nan:.4f}"]
            } for band in bands}
            
            # Add envelope models to correlation data
            for cl in context_lengths:
                if cl in envelope_results:
                    correlation_data["Model"].append(f"Envelope Context {cl:.2f}s")
                    correlation_data["Mean Correlation"].append(f"{envelope_results[cl]['avg_correlation']:.4f}")
                    correlation_data["Std Dev"].append(f"{np.std(envelope_results[cl]['correlations']):.4f}")
                    correlation_data["Min"].append(f"{np.min(envelope_results[cl]['correlations']):.4f}")
                    correlation_data["Max"].append(f"{np.max(envelope_results[cl]['correlations']):.4f}")
                    
                    # Add band-specific data
                    for band in bands:
                        band_correlations = envelope_results[cl]['band_correlations'][band]
                        band_correlation_data[band]["Model"].append(f"Envelope Context {cl:.2f}s")
                        band_correlation_data[band]["Mean Correlation"].append(f"{np.mean([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Std Dev"].append(f"{np.std([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Min"].append(f"{np.min([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Max"].append(f"{np.max([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
            
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
                        band_correlation_data[band]["Mean Correlation"].append(f"{np.mean([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Std Dev"].append(f"{np.std([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Min"].append(f"{np.min([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Max"].append(f"{np.max([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
            
            # Add frozen EEG models to correlation data
            for cl in context_lengths:
                if cl in frozen_eeg_results:
                    correlation_data["Model"].append(f"Frozen EEG Context {cl:.2f}s")
                    correlation_data["Mean Correlation"].append(f"{frozen_eeg_results[cl]['avg_correlation']:.4f}")
                    correlation_data["Std Dev"].append(f"{np.std(frozen_eeg_results[cl]['correlations']):.4f}")
                    correlation_data["Min"].append(f"{np.min(frozen_eeg_results[cl]['correlations']):.4f}")
                    correlation_data["Max"].append(f"{np.max(frozen_eeg_results[cl]['correlations']):.4f}")
                    
                    # Add band-specific data
                    for band in bands:
                        band_correlations = frozen_eeg_results[cl]['band_correlations'][band]
                        band_correlation_data[band]["Model"].append(f"Frozen EEG Context {cl:.2f}s")
                        band_correlation_data[band]["Mean Correlation"].append(f"{np.mean([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Std Dev"].append(f"{np.std([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Min"].append(f"{np.min([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
                        band_correlation_data[band]["Max"].append(f"{np.max([corr for corr in band_correlations if not np.isnan(corr)]) if band_correlations else np.nan:.4f}")
            
            # Save correlation statistics to a text file
            with open(os.path.join(subject_output_dir, f"{subject_id}_correlation_statistics.txt"), 'w') as f:
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
        all_subjects_psd_data["envelope"][context_length] = [psd for psd in psd_predictions]
        all_subjects_psd_data["fusion"][context_length] = [psd for psd in psd_predictions]
        all_subjects_psd_data["frozen_eeg"][context_length] = [psd for psd in psd_predictions]
        
        all_subjects_correlation_data["eeg_only"].append(eeg_only_correlations)
        all_subjects_correlation_data["envelope"][context_length] = correlations
        all_subjects_correlation_data["fusion"][context_length] = correlations
        all_subjects_correlation_data["frozen_eeg"][context_length] = correlations
        
        all_subjects_band_correlation_data["eeg_only"][band] += band_correlations
        all_subjects_band_correlation_data["envelope"][band][context_length] = band_correlations
        all_subjects_band_correlation_data["fusion"][band][context_length] = band_correlations
        all_subjects_band_correlation_data["frozen_eeg"][band][context_length] = band_correlations
    
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
        
        # Plot envelope models
        for context_length in context_lengths:
            if context_length in all_subjects_psd_data["envelope"]:
                avg_psd = np.mean([psd for psd in all_subjects_psd_data["envelope"][context_length]], axis=0)
                plt.semilogy(freq_values, avg_psd, '-', 
                            color=envelope_cmap[context_lengths.index(context_length)], 
                            label=f"Envelope {context_length:.2f}s")
        
        # Plot fusion models
        for context_length in context_lengths:
            if context_length in all_subjects_psd_data["fusion"]:
                avg_psd = np.mean([psd for psd in all_subjects_psd_data["fusion"][context_length]], axis=0)
                plt.semilogy(freq_values, avg_psd, '-', 
                            color=fusion_cmap[context_lengths.index(context_length)], 
                            label=f"Fusion {context_length:.2f}s")
        
        # Plot frozen EEG models
        for context_length in context_lengths:
            if context_length in all_subjects_psd_data["frozen_eeg"]:
                avg_psd = np.mean([psd for psd in all_subjects_psd_data["frozen_eeg"][context_length]], axis=0)
                plt.semilogy(freq_values, avg_psd, '-', 
                            color=frozen_eeg_cmap[context_lengths.index(context_length)], 
                            label=f"Frozen EEG {context_length:.2f}s")
        
        plt.title("Average Power Spectral Density Across All Subjects")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (log scale)")
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_psd.png"), dpi=plot_dpi)
        plt.close()
        
        # Average Correlation Plot
        plt.figure(figsize=(10, 6))
        
        # Calculate averages for each model type
        avg_eeg_only_corr = np.mean(all_subjects_correlation_data["eeg_only"])
        
        # Plot correlation means with error bars
        for context_length in context_lengths:
            # Envelope models
            if context_length in all_subjects_correlation_data["envelope"]:
                corrs = all_subjects_correlation_data["envelope"][context_length]
                plt.errorbar(context_length, np.mean(corrs), yerr=np.std(corrs), fmt='o-', 
                            capsize=5, markersize=8, linewidth=2, color=envelope_color, 
                            label="Envelope" if context_length == context_lengths[0] else "")
            
            # Fusion models
            if context_length in all_subjects_correlation_data["fusion"]:
                corrs = all_subjects_correlation_data["fusion"][context_length]
                plt.errorbar(context_length, np.mean(corrs), yerr=np.std(corrs), fmt='o-', 
                            capsize=5, markersize=8, linewidth=2, color=fusion_color, 
                            label="Fusion" if context_length == context_lengths[0] else "")
            
            # Frozen EEG models
            if context_length in all_subjects_correlation_data["frozen_eeg"]:
                corrs = all_subjects_correlation_data["frozen_eeg"][context_length]
                plt.errorbar(context_length, np.mean(corrs), yerr=np.std(corrs), fmt='o-', 
                            capsize=5, markersize=8, linewidth=2, color=frozen_eeg_color, 
                            label="Frozen EEG" if context_length == context_lengths[0] else "")
        
        # Add horizontal line for EEG-only model
        plt.axhline(y=avg_eeg_only_corr, color='r', linestyle='--', linewidth=2,
                   label=f"Happyquoka (EEG-only): {avg_eeg_only_corr:.4f}")
        
        plt.title("Average Correlation with Ground Truth vs Context Length (All Subjects)")
        plt.xlabel("Context Length (seconds)")
        plt.ylabel("Pearson Correlation")
        plt.grid(True, ls="--")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_correlation.png"), dpi=plot_dpi)
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
        plt.bar(bar_positions - 1.5*bar_width, eeg_only_band_means, bar_width, 
               yerr=eeg_only_band_stds, color='r', alpha=0.7, 
               label=f"Happyquoka ({avg_eeg_only_corr:.3f})")
        
        # Find best context length for each model type
        best_envelope_context = max(all_subjects_correlation_data["envelope"].keys(), 
                                  key=lambda k: np.mean(all_subjects_correlation_data["envelope"][k]))
        best_fusion_context = max(all_subjects_correlation_data["fusion"].keys(), 
                                 key=lambda k: np.mean(all_subjects_correlation_data["fusion"][k]))
        
        # Calculate and plot envelope band means
        envelope_band_means = []
        envelope_band_stds = []
        for band in bands:
            corrs = all_subjects_band_correlation_data["envelope"][band][best_envelope_context]
            envelope_band_means.append(np.mean(corrs))
            envelope_band_stds.append(np.std(corrs))
        
        plt.bar(bar_positions - 0.5*bar_width, envelope_band_means, bar_width, 
               yerr=envelope_band_stds, color=envelope_color, alpha=0.7,
               label=f"Envelope {best_envelope_context:.2f}s")
        
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
            plt.text(bar_positions[i] - 1.5*bar_width, eeg_only_band_means[i] + 0.01, 
                    f"{eeg_only_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
            
            # Envelope
            plt.text(bar_positions[i] - 0.5*bar_width, envelope_band_means[i] + 0.01, 
                    f"{envelope_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
            
            # Fusion
            plt.text(bar_positions[i] + 0.5*bar_width, fusion_band_means[i] + 0.01, 
                    f"{fusion_band_means[i]:.3f}", ha='center', va='bottom', fontsize=7, color='black', rotation=90)
        
        plt.title("Average Correlation by Frequency Band (All Subjects)")
        plt.xlabel("Frequency Band")
        plt.ylabel("Pearson Correlation")
        plt.xticks(bar_positions, bands)
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(avg_plots_dir, "average_band_correlation_bars.png"), dpi=plot_dpi)
        plt.close()
        
        # Save average statistics to a text file
        with open(os.path.join(avg_plots_dir, "average_correlation_statistics.txt"), 'w') as f:
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
            
            # Write envelope model statistics
            for context_length in context_lengths:
                if context_length in all_subjects_correlation_data["envelope"]:
                    corrs = all_subjects_correlation_data["envelope"][context_length]
                    f.write(f"{f'Envelope {context_length:.2f}s':<20} "
                            f"{np.mean(corrs):.4f:<20} "
                            f"{np.std(corrs):.4f:<15} "
                            f"{np.min(corrs):.4f:<15} "
                            f"{np.max(corrs):.4f:<15}\n")
            
            # Write fusion model statistics
            for context_length in context_lengths:
                if context_length in all_subjects_correlation_data["fusion"]:
                    corrs = all_subjects_correlation_data["fusion"][context_length]
                    f.write(f"{f'Fusion {context_length:.2f}s':<20} "
                            f"{np.mean(corrs):.4f:<20} "
                            f"{np.std(corrs):.4f:<15} "
                            f"{np.min(corrs):.4f:<15} "
                            f"{np.max(corrs):.4f:<15}\n")
            
            # Write frozen EEG model statistics
            for context_length in context_lengths:
                if context_length in all_subjects_correlation_data["frozen_eeg"]:
                    corrs = all_subjects_correlation_data["frozen_eeg"][context_length]
                    f.write(f"{f'Frozen EEG {context_length:.2f}s':<20} "
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
                
                # Envelope band statistics
                for context_length in context_lengths:
                    if context_length in all_subjects_band_correlation_data["envelope"][band]:
                        corrs = all_subjects_band_correlation_data["envelope"][band][context_length]
                        f.write(f"{f'Envelope {context_length:.2f}s':<20} "
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
                
                # Frozen EEG band statistics
                for context_length in context_lengths:
                    if context_length in all_subjects_band_correlation_data["frozen_eeg"][band]:
                        corrs = all_subjects_band_correlation_data["frozen_eeg"][band][context_length]
                        f.write(f"{f'Frozen EEG {context_length:.2f}s':<20} "
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
    compare_envelope_models_for_all_subjects() 