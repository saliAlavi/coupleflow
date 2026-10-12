#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from organize_models import organize_model_paths

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

def filter_subject_data(data, subject_id="sub-001"):
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

def compare_envelope_models_for_sub001():
    """Compare envelope context models with different context lengths for subject sub-001"""
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
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
    
    print("Step 2/5: Loading and filtering EEG-only model for ground truth...")
    
    # Load and filter the EEG-only model first to get ground truth
    with open(eeg_only_model, 'rb') as file_handle:
        eeg_only_data = pickle.load(file_handle)
    
    # Filter for subject sub-001
    sub001_eeg_only = filter_subject_data(eeg_only_data)
    
    # Extract ground truths
    ground_truths = sub001_eeg_only["ground_truths"]
    print(f"Found {len(ground_truths)} segments for subject sub-001")
    
    # Extract Happyquoka (EEG-only) model predictions
    eeg_only_predictions = [output["envelope"] for output in sub001_eeg_only["outputs"]]
    
    # Compute PSD for ground truth
    print("Computing PSDs for ground truth...")
    psd_ground_truths, freq_values = compute_psd_welch(ground_truths, 64)
    print(f"Frequency values type: {type(freq_values)}, shape: {freq_values.shape}")
    
    # Compute PSD for Happyquoka model
    print("Computing PSDs for Happyquoka (EEG-only) model...")
    psd_eeg_only, _ = compute_psd_welch(eeg_only_predictions, 64)
    
    avg_psd_ground_truth = np.mean(psd_ground_truths, axis=0)
    avg_psd_eeg_only = np.mean(psd_eeg_only, axis=0)
    print(f"Average ground truth PSD shape: {avg_psd_ground_truth.shape}")
    
    # Calculate correlation for Happyquoka model
    eeg_only_correlations = []
    for i in range(len(ground_truths)):
        gt = ground_truths[i].reshape(-1)
        pred = eeg_only_predictions[i].reshape(-1)
        corr = np.corrcoef(gt, pred)[0, 1]
        eeg_only_correlations.append(corr)
    
    avg_eeg_only_correlation = np.mean(eeg_only_correlations)
    print(f"Happyquoka (EEG-only) model average correlation: {avg_eeg_only_correlation:.4f}")
    
    # Process, load and gather all context lengths across models
    all_context_lengths = set()
    
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
    print("\nStep 3/5: Processing envelope and fusion models with different context lengths...")
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
            
        print(f"Processing envelope model with context length {context_length:.2f}s")
        
        # Load model data
        with open(model_path, 'rb') as file_handle:
            model_data = pickle.load(file_handle)
        
        # Filter for subject sub-001
        sub001_data = filter_subject_data(model_data)
        
        # Extract predictions
        predictions = [output["envelope"] for output in sub001_data["outputs"]]
        
        # Compute PSD
        psd_predictions, _ = compute_psd_welch(predictions, 64)
        avg_psd = np.mean(psd_predictions, axis=0)
        
        # Calculate correlations with ground truth
        correlations = []
        for i in range(len(ground_truths)):
            gt = ground_truths[i].reshape(-1)
            pred = predictions[i].reshape(-1)
            corr = np.corrcoef(gt, pred)[0, 1]
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # Store results
        envelope_results[context_length] = {
            "predictions": predictions,
            "psd": psd_predictions,
            "avg_psd": avg_psd,
            "correlations": correlations,
            "avg_correlation": avg_correlation
        }
    
    # Process fusion models
    for _, row in fusion_models.iterrows():
        model_path = row["path"]
        context_length = row["context_length"]
        
        # Skip if not in common context lengths
        if context_length not in context_lengths:
            continue
            
        print(f"Processing fusion model with context length {context_length:.2f}s")
        
        # Load model data
        with open(model_path, 'rb') as file_handle:
            model_data = pickle.load(file_handle)
        
        # Filter for subject sub-001
        sub001_data = filter_subject_data(model_data)
        
        # Extract predictions
        predictions = [output["envelope"] for output in sub001_data["outputs"]]
        
        # Compute PSD
        psd_predictions, _ = compute_psd_welch(predictions, 64)
        avg_psd = np.mean(psd_predictions, axis=0)
        
        # Calculate correlations with ground truth
        correlations = []
        for i in range(len(ground_truths)):
            gt = ground_truths[i].reshape(-1)
            pred = predictions[i].reshape(-1)
            corr = np.corrcoef(gt, pred)[0, 1]
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # Store results
        fusion_results[context_length] = {
            "predictions": predictions,
            "psd": psd_predictions,
            "avg_psd": avg_psd,
            "correlations": correlations,
            "avg_correlation": avg_correlation
        }
    
    # Process frozen EEG models
    for _, row in frozen_eeg_models.iterrows():
        model_path = row["path"]
        context_length = row["context_length"]
        
        # Skip if not in common context lengths
        if context_length not in context_lengths:
            continue
            
        print(f"Processing frozen EEG model with context length {context_length:.2f}s")
        
        # Load model data
        with open(model_path, 'rb') as file_handle:
            model_data = pickle.load(file_handle)
        
        # Filter for subject sub-001
        sub001_data = filter_subject_data(model_data)
        
        # Extract predictions
        predictions = [output["envelope"] for output in sub001_data["outputs"]]
        
        # Compute PSD
        psd_predictions, _ = compute_psd_welch(predictions, 64)
        avg_psd = np.mean(psd_predictions, axis=0)
        
        # Calculate correlations with ground truth
        correlations = []
        for i in range(len(ground_truths)):
            gt = ground_truths[i].reshape(-1)
            pred = predictions[i].reshape(-1)
            corr = np.corrcoef(gt, pred)[0, 1]
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # Store results
        frozen_eeg_results[context_length] = {
            "predictions": predictions,
            "psd": psd_predictions,
            "avg_psd": avg_psd,
            "correlations": correlations,
            "avg_correlation": avg_correlation
        }
    
    # Use color gradient for different context lengths
    envelope_colors = plt.cm.viridis(np.linspace(0, 1, len(context_lengths)))
    fusion_colors = plt.cm.plasma(np.linspace(0, 1, len(context_lengths)))
    frozen_eeg_colors = plt.cm.cool(np.linspace(0, 1, len(context_lengths)))  # Cool colormap (cyan to magenta)
    
    print("\nStep 4/5: Generating plots...")
    
    try:
        # Plot average PSDs
        print("Generating average PSD plot...")
        plt.figure(figsize=(12, 8))
        
        # Create better color schemes for three model types
        envelope_cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(context_lengths)))
        fusion_cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(context_lengths)))
        frozen_eeg_cmap = plt.cm.Purples(np.linspace(0.4, 0.9, len(context_lengths)))  # Purple for frozen EEG
        
        plt.semilogy(freq_values, avg_psd_ground_truth, 'k-', linewidth=2, label="Ground Truth")
        plt.semilogy(freq_values, avg_psd_eeg_only, 'r-', linewidth=2, label="Happyquoka (EEG-only)")
        
        for i, context_length in enumerate(context_lengths):
            plt.semilogy(freq_values, envelope_results[context_length]["avg_psd"], '-', 
                        color=envelope_cmap[i], label=f"Envelope {context_length:.2f}s")
            plt.semilogy(freq_values, fusion_results[context_length]["avg_psd"], '-', 
                        color=fusion_cmap[i], label=f"Fusion {context_length:.2f}s")
            if context_length in frozen_eeg_results:
                plt.semilogy(freq_values, frozen_eeg_results[context_length]["avg_psd"], '-', 
                            color=frozen_eeg_cmap[i], label=f"Frozen EEG {context_length:.2f}s")
        
        plt.title("Average Power Spectral Density for Different Context Lengths (sub-001)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (log scale)")
        
        # Group similar model types together in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Re-order labels to group by model type
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
        
        plt.legend(handles, labels, loc='best', fontsize=9)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig("plots/sub001_context_comparison_psd.png", dpi=plot_dpi)
        plt.close()
        
        # Plot correlation vs context length
        print("Generating correlation vs context length plot...")
        plt.figure(figsize=(10, 6))
        
        # Create better color schemes for three model types - using darker colors for line plots
        envelope_color = plt.cm.Blues(0.7)  # Use a single darker blue for envelope line
        fusion_color = plt.cm.Oranges(0.7)  # Use a single darker orange for fusion line
        frozen_eeg_color = plt.cm.Purples(0.7)  # Use a single darker purple for frozen EEG line
        
        envelope_correlation_means = [envelope_results[cl]["avg_correlation"] for cl in context_lengths]
        envelope_correlation_stds = [np.std(envelope_results[cl]["correlations"]) for cl in context_lengths]
        fusion_correlation_means = [fusion_results[cl]["avg_correlation"] for cl in context_lengths]
        fusion_correlation_stds = [np.std(fusion_results[cl]["correlations"]) for cl in context_lengths]
        
        # Prepare frozen EEG data if available for all context lengths
        frozen_eeg_correlation_means = []
        frozen_eeg_correlation_stds = []
        for cl in context_lengths:
            if cl in frozen_eeg_results:
                frozen_eeg_correlation_means.append(frozen_eeg_results[cl]["avg_correlation"])
                frozen_eeg_correlation_stds.append(np.std(frozen_eeg_results[cl]["correlations"]))
            else:
                frozen_eeg_correlation_means.append(np.nan)
                frozen_eeg_correlation_stds.append(np.nan)
        
        plt.errorbar(context_lengths, envelope_correlation_means, yerr=envelope_correlation_stds, fmt='o-', 
                    capsize=5, markersize=8, linewidth=2, color=envelope_color, label="Envelope")
        plt.errorbar(context_lengths, fusion_correlation_means, yerr=fusion_correlation_stds, fmt='o-', 
                    capsize=5, markersize=8, linewidth=2, color=fusion_color, label="Fusion")
        
        # Plot frozen EEG correlation means if they exist
        plt.errorbar(context_lengths, frozen_eeg_correlation_means, yerr=frozen_eeg_correlation_stds, fmt='o-', 
                    capsize=5, markersize=8, linewidth=2, color=frozen_eeg_color, label="Frozen EEG")
        
        # Add horizontal line for EEG-only (Happyquoka) model
        plt.axhline(y=avg_eeg_only_correlation, color='r', linestyle='--', linewidth=2,
                   label=f"Happyquoka (EEG-only): {avg_eeg_only_correlation:.4f}")
        
        plt.title("Average Correlation with Ground Truth vs Context Length (sub-001)")
        plt.xlabel("Context Length (seconds)")
        plt.ylabel("Pearson Correlation")
        
        # Add data point values as text labels
        for i, cl in enumerate(context_lengths):
            # Envelope model labels (slightly offset to avoid overlap)
            plt.text(cl-0.02, envelope_correlation_means[i] - 0.015, f"{envelope_correlation_means[i]:.3f}", 
                    ha='right', va='top', fontsize=7, color=envelope_color)
            
            # Fusion model labels
            plt.text(cl, fusion_correlation_means[i] + 0.01, f"{fusion_correlation_means[i]:.3f}", 
                    ha='center', va='bottom', fontsize=7, color=fusion_color)
            
            # Frozen EEG model labels (slightly offset to avoid overlap)
            if not np.isnan(frozen_eeg_correlation_means[i]):
                plt.text(cl+0.02, frozen_eeg_correlation_means[i] - 0.015, f"{frozen_eeg_correlation_means[i]:.3f}", 
                        ha='left', va='top', fontsize=7, color=frozen_eeg_color)
        
        plt.grid(True, ls="--")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("plots/sub001_context_correlation.png", dpi=plot_dpi)
        plt.close()
        
        # Example time series plot for a single segment with different context lengths
        print("Generating example time series plot...")
        example_idx = 0
        plt.figure(figsize=(14, 8))
        
        # Create better color schemes for three model types
        envelope_cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(context_lengths)))
        fusion_cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(context_lengths)))
        frozen_eeg_cmap = plt.cm.Purples(np.linspace(0.4, 0.9, len(context_lengths)))  # Purple for frozen EEG
        
        # Plot ground truth
        plt.plot(normalize(ground_truths[example_idx].reshape(-1)), 'k-', linewidth=2, label="Ground Truth")
        
        # Plot Happyquoka (EEG-only) prediction
        plt.plot(normalize(eeg_only_predictions[example_idx].reshape(-1)), 'r-', linewidth=2, 
                label="Happyquoka (EEG-only)")
        
        # Plot each model prediction
        for i, context_length in enumerate(context_lengths):
            if context_length in envelope_results:
                predictions = envelope_results[context_length]["predictions"]
                plt.plot(normalize(predictions[example_idx].reshape(-1)), '-', color=envelope_cmap[i], 
                        label=f"Envelope {context_length:.2f}s")
            
            if context_length in fusion_results:
                predictions = fusion_results[context_length]["predictions"]
                plt.plot(normalize(predictions[example_idx].reshape(-1)), '-', color=fusion_cmap[i], 
                        label=f"Fusion {context_length:.2f}s")
            
            if context_length in frozen_eeg_results:
                predictions = frozen_eeg_results[context_length]["predictions"]
                plt.plot(normalize(predictions[example_idx].reshape(-1)), '-', color=frozen_eeg_cmap[i], 
                        label=f"Frozen EEG {context_length:.2f}s")
        
        plt.title(f"Example Time Series Comparison for Different Context Lengths (sub-001, Segment {example_idx})")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized Amplitude")
        
        # Group similar model types together in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Re-order labels to group by model type
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
        
        plt.legend(handles, labels, loc='upper right', fontsize=9)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/sub001_context_timeseries.png", dpi=plot_dpi)
        plt.close()
        
        # Plot comparison of PSDs at different frequency bands
        print("Generating frequency band power comparison plot...")
        plt.figure(figsize=(14, 9))  # Increase figure size for better readability
        
        # Define frequency bands
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 32),
        }
        
        # Calculate average PSD in each band for each model
        band_powers = {band: [] for band in bands}
        
        # Ground truth band powers
        gt_band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            idx = np.logical_and(freq_values >= f_low, freq_values <= f_high)
            gt_band_powers[band_name] = np.mean(avg_psd_ground_truth[idx])
        
        # Happyquoka (EEG-only) band powers
        eeg_only_band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            idx = np.logical_and(freq_values >= f_low, freq_values <= f_high)
            eeg_only_band_powers[band_name] = np.mean(avg_psd_eeg_only[idx])
        
        # Calculate band powers for envelope models
        envelope_band_powers = {band: [] for band in bands}
        for context_length in context_lengths:
            model_avg_psd = envelope_results[context_length]["avg_psd"]
            for band_name, (f_low, f_high) in bands.items():
                idx = np.logical_and(freq_values >= f_low, freq_values <= f_high)
                envelope_band_powers[band_name].append(np.mean(model_avg_psd[idx]))
        
        # Calculate band powers for fusion models
        fusion_band_powers = {band: [] for band in bands}
        for context_length in context_lengths:
            model_avg_psd = fusion_results[context_length]["avg_psd"]
            for band_name, (f_low, f_high) in bands.items():
                idx = np.logical_and(freq_values >= f_low, freq_values <= f_high)
                fusion_band_powers[band_name].append(np.mean(model_avg_psd[idx]))
        
        # Calculate band powers for frozen EEG models
        frozen_eeg_band_powers = {band: [] for band in bands}
        for context_length in context_lengths:
            if context_length in frozen_eeg_results:
                model_avg_psd = frozen_eeg_results[context_length]["avg_psd"]
                for band_name, (f_low, f_high) in bands.items():
                    idx = np.logical_and(freq_values >= f_low, freq_values <= f_high)
                    frozen_eeg_band_powers[band_name].append(np.mean(model_avg_psd[idx]))
            else:
                # Add placeholder if this context length doesn't exist for frozen EEG models
                for band_name in bands:
                    frozen_eeg_band_powers[band_name].append(np.nan)
        
        # Plot band powers with grouped bars
        x = np.arange(len(bands))
        n_context_lengths = len(context_lengths)
        bar_width = 0.8 / (3 * n_context_lengths + 2)  # +2 for ground truth and EEG-only, x3 for 3 model types
        
        # Create figure with logarithmic y-scale
        plt.figure(figsize=(16, 9))  # Increased width for more models
        plt.yscale('log')
        
        # Create better color schemes for three model types
        envelope_cmap = plt.cm.Blues(np.linspace(0.4, 0.9, n_context_lengths))
        fusion_cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, n_context_lengths))
        frozen_eeg_cmap = plt.cm.Purples(np.linspace(0.4, 0.9, n_context_lengths))  # Purple for frozen EEG
        
        # Plot ground truth band powers (black)
        plt.bar(x - 0.4, [gt_band_powers[band] for band in bands], bar_width, color='k', label="Ground Truth")
        
        # Plot EEG-only band powers (red)
        plt.bar(x - 0.4 + bar_width, [eeg_only_band_powers[band] for band in bands], bar_width, color='r', 
               label="Happyquoka (EEG-only)")
        
        # Spacing between model groups
        spacing = bar_width * 0.5
        
        # Plot envelope models (grouped together)
        for i, context_length in enumerate(context_lengths):
            offset = -0.4 + (i + 2) * bar_width
            plt.bar(x + offset, [envelope_band_powers[band][i] for band in bands], bar_width, 
                   color=envelope_cmap[i], label=f"Envelope {context_length:.2f}s")
        
        # Plot fusion models (grouped together)
        for i, context_length in enumerate(context_lengths):
            offset = -0.4 + (i + 2 + n_context_lengths) * bar_width + spacing
            plt.bar(x + offset, [fusion_band_powers[band][i] for band in bands], bar_width, 
                   color=fusion_cmap[i], label=f"Fusion {context_length:.2f}s")
        
        # Plot frozen EEG models (grouped together)
        for i, context_length in enumerate(context_lengths):
            if not np.isnan(list(frozen_eeg_band_powers.values())[0][i]):  # Check if we have data for this context length
                offset = -0.4 + (i + 2 + 2 * n_context_lengths) * bar_width + 2 * spacing
                plt.bar(x + offset, [frozen_eeg_band_powers[band][i] for band in bands], bar_width, 
                       color=frozen_eeg_cmap[i], label=f"Frozen EEG {context_length:.2f}s")
        
        # Create x-axis labels with frequency ranges
        x_labels = [f"{band_name.capitalize()}\n({f_low}-{f_high} Hz)" for band_name, (f_low, f_high) in bands.items()]
        
        # Add divider lines between model groups
        for band_idx in range(len(bands)):
            # Vertical line between ground truth/EEG-only and envelope models
            plt.axvline(x=band_idx - 0.4 + 2*bar_width + bar_width/2, 
                      ymin=0, ymax=0.01, color='gray', linestyle='-', linewidth=1)
            
            # Vertical line between envelope and fusion models
            plt.axvline(x=band_idx - 0.4 + (2+n_context_lengths)*bar_width + spacing/2, 
                      ymin=0, ymax=0.01, color='gray', linestyle='-', linewidth=1)
            
            # Vertical line between fusion and frozen EEG models
            plt.axvline(x=band_idx - 0.4 + (2+2*n_context_lengths)*bar_width + 1.5*spacing, 
                      ymin=0, ymax=0.01, color='gray', linestyle='-', linewidth=1)
        
        plt.title("Average PSD by Frequency Band for Different Context Lengths (sub-001)", fontsize=14)
        plt.xlabel("Frequency Band", fontsize=12)
        plt.ylabel("Average Power (log scale)", fontsize=12)
        plt.xticks(x, x_labels, fontsize=11)
        plt.yticks(fontsize=10)
        
        # Add a grid for better readability
        plt.grid(which='both', axis='y', linestyle='--', alpha=0.7)
        plt.grid(which='major', axis='x', linestyle='-', alpha=0.3)
        
        # Add model type annotations
        for band_idx in range(len(bands)):
            # Ground truth/EEG-only annotation
            gt_center = band_idx - 0.4 + bar_width/2
            plt.annotate("Reference", xy=(gt_center, 1e-8), xytext=(gt_center, 1e-9),
                      ha='center', rotation=90, fontsize=8, color='gray',
                      arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            
            # Envelope models annotation
            env_center = band_idx - 0.4 + (2 + n_context_lengths/2) * bar_width
            plt.annotate("Envelope", xy=(env_center, 1e-8), xytext=(env_center, 1e-9),
                      ha='center', rotation=90, fontsize=8, color='blue',
                      arrowprops=dict(arrowstyle='-', color='blue', lw=0.5))
            
            # Fusion models annotation
            fusion_center = band_idx - 0.4 + (2 + n_context_lengths + n_context_lengths/2) * bar_width + spacing
            plt.annotate("Fusion", xy=(fusion_center, 1e-8), xytext=(fusion_center, 1e-9),
                      ha='center', rotation=90, fontsize=8, color='orangered',
                      arrowprops=dict(arrowstyle='-', color='orangered', lw=0.5))
            
            # Frozen EEG models annotation
            frozen_eeg_center = band_idx - 0.4 + (2 + 2*n_context_lengths + n_context_lengths/2) * bar_width + 2*spacing
            plt.annotate("Frozen EEG", xy=(frozen_eeg_center, 1e-8), xytext=(frozen_eeg_center, 1e-9),
                      ha='center', rotation=90, fontsize=8, color='purple',
                      arrowprops=dict(arrowstyle='-', color='purple', lw=0.5))
        
        # Place legend outside to the right, with a better grouping structure
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Create color-coded sections in the legend
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), 
                 borderaxespad=0, fontsize=8, framealpha=0.8)
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.78, 1])  # Adjust the right margin to accommodate more legend items
        
        plt.savefig("plots/sub001_context_band_powers.png", dpi=plot_dpi)
        plt.close()
    
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create table with correlation statistics
    print("\nStep 5/5: Calculating and displaying correlation statistics...")
    
    # Prepare data for frozen EEG models
    frozen_eeg_means = []
    frozen_eeg_stds = []
    frozen_eeg_mins = []
    frozen_eeg_maxs = []
    frozen_eeg_labels = []
    
    for cl in context_lengths:
        if cl in frozen_eeg_results:
            frozen_eeg_labels.append(f"Frozen EEG Context {cl:.2f}s")
            frozen_eeg_means.append(f"{frozen_eeg_results[cl]['avg_correlation']:.4f}")
            frozen_eeg_stds.append(f"{np.std(frozen_eeg_results[cl]['correlations']):.4f}")
            frozen_eeg_mins.append(f"{np.min(frozen_eeg_results[cl]['correlations']):.4f}")
            frozen_eeg_maxs.append(f"{np.max(frozen_eeg_results[cl]['correlations']):.4f}")
    
    correlation_data = {
        "Model": ["Happyquoka (EEG-only)"] + 
                [f"Envelope Context {cl:.2f}s" for cl in context_lengths] + 
                [f"Fusion Context {cl:.2f}s" for cl in context_lengths] +
                frozen_eeg_labels,
        "Mean Correlation": [f"{avg_eeg_only_correlation:.4f}"] + 
                           [f"{envelope_results[cl]['avg_correlation']:.4f}" for cl in context_lengths] +
                           [f"{fusion_results[cl]['avg_correlation']:.4f}" for cl in context_lengths] +
                           frozen_eeg_means,
        "Std Dev": [f"{np.std(eeg_only_correlations):.4f}"] + 
                  [f"{np.std(envelope_results[cl]['correlations']):.4f}" for cl in context_lengths] +
                  [f"{np.std(fusion_results[cl]['correlations']):.4f}" for cl in context_lengths] +
                  frozen_eeg_stds,
        "Min": [f"{np.min(eeg_only_correlations):.4f}"] + 
              [f"{np.min(envelope_results[cl]['correlations']):.4f}" for cl in context_lengths] +
              [f"{np.min(fusion_results[cl]['correlations']):.4f}" for cl in context_lengths] +
              frozen_eeg_mins,
        "Max": [f"{np.max(eeg_only_correlations):.4f}"] + 
              [f"{np.max(envelope_results[cl]['correlations']):.4f}" for cl in context_lengths] +
              [f"{np.max(fusion_results[cl]['correlations']):.4f}" for cl in context_lengths] +
              frozen_eeg_maxs
    }
    
    print("\nCorrelation Statistics by Model:")
    print("-" * 90)
    print(f"{'Model':<20} {'Mean Correlation':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 90)
    
    for i in range(len(correlation_data["Model"])):
        print(f"{correlation_data['Model'][i]:<20} "
              f"{correlation_data['Mean Correlation'][i]:<20} "
              f"{correlation_data['Std Dev'][i]:<15} "
              f"{correlation_data['Min'][i]:<15} "
              f"{correlation_data['Max'][i]:<15}")
    
    print("\nPlots saved to plots/ directory")

if __name__ == "__main__":
    compare_envelope_models_for_sub001() 