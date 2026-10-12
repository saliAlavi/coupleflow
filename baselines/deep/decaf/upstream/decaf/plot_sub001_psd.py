#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Model paths
eeg_only_model = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec/model_data_Happyquoka4_pearson_l1_happyquoka4_20250314_230318.pkl"
envelope_0p5_model = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec/model_data_TwoBranchFrozenEegContext0.5SecEnvelopeOnly_pearson_l1_two_branch_frozen_eeg_0sec_envelope_only_20250317_045537.pkl"
fusion_frozeneeg_0p5_model = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec/model_data_TwoBranchFrozenEegContext0.5SecDynamic_pearson_l1_two_branch_frozen_eeg_0sec_dynamic_20250317_023552.pkl"
fusion_0p5_model = "checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec/model_data_TwoBranchContext0.5SecDynamic_pearson_l1_two_branch_context_0.5sec_dynamic_20250320_181900.pkl"

# Load pickle files
with open(eeg_only_model, 'rb') as f:
    eeg_only_data = pickle.load(f)

with open(envelope_0p5_model, 'rb') as f:
    envelope_0p5_data = pickle.load(f)

with open(fusion_frozeneeg_0p5_model, 'rb') as f:
    fusion_frozeneeg_0p5_data = pickle.load(f)

with open(fusion_0p5_model, 'rb') as f:
    fusion_0p5_data = pickle.load(f)

def compute_psd_welch(list_of_arrays, sampling_rate):
    """
    Compute the power spectral density using Welch's method for a list of arrays.
    
    Args:
        list_of_arrays: List of arrays to compute PSD for
        sampling_rate: Sampling rate in Hz
    
    Returns:
        psd_list: List of PSDs
        f: Frequency bins
    """
    psd_list = []
    for x in list_of_arrays:
        f, Pxx = signal.welch(x.reshape(-1), sampling_rate, nperseg=64, noverlap=32)
        psd_list.append(Pxx)
    return psd_list, f

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
        "outputs": [data["outputs"][i] for i in filtered_indices]
    }
    
    return filtered_data

# Filter data for sub-001
sub001_eeg_only = filter_subject_data(eeg_only_data)
sub001_envelope_0p5 = filter_subject_data(envelope_0p5_data)
sub001_fusion_frozeneeg_0p5 = filter_subject_data(fusion_frozeneeg_0p5_data)
sub001_fusion_0p5 = filter_subject_data(fusion_0p5_data)

print(f"Found {len(sub001_eeg_only['metadata'])} segments for subject sub-001")

# Extract ground truths and predictions
ground_truths = sub001_eeg_only["ground_truths"]

# Extract model predictions
predictions_eeg_only = [output["envelope"] for output in sub001_eeg_only["outputs"]]
predictions_envelope = [output["envelope"] for output in sub001_envelope_0p5["outputs"]]
predictions_fusion_frozeneeg = [output["envelope"] for output in sub001_fusion_frozeneeg_0p5["outputs"]]
predictions_fusion = [output["envelope"] for output in sub001_fusion_0p5["outputs"]]

# Compute PSDs
psd_ground_truths, f = compute_psd_welch(ground_truths, 64)
psd_eeg_only, _ = compute_psd_welch(predictions_eeg_only, 64)
psd_envelope, _ = compute_psd_welch(predictions_envelope, 64)
psd_fusion_frozeneeg, _ = compute_psd_welch(predictions_fusion_frozeneeg, 64)
psd_fusion, _ = compute_psd_welch(predictions_fusion, 64)

# Average PSDs
avg_psd_ground_truths = np.mean(psd_ground_truths, axis=0)
avg_psd_eeg_only = np.mean(psd_eeg_only, axis=0)
avg_psd_envelope = np.mean(psd_envelope, axis=0)
avg_psd_fusion_frozeneeg = np.mean(psd_fusion_frozeneeg, axis=0)
avg_psd_fusion = np.mean(psd_fusion, axis=0)

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Plot average PSDs on log scale
plt.figure(figsize=(10, 6))
plt.semilogy(f, avg_psd_ground_truths, 'k-', linewidth=2)
plt.semilogy(f, avg_psd_eeg_only, 'b-')
plt.semilogy(f, avg_psd_envelope, 'g-')
plt.semilogy(f, avg_psd_fusion_frozeneeg, 'r-')
plt.semilogy(f, avg_psd_fusion, 'm-')

plt.title("Average Power Spectral Density for Subject sub-001")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (log scale)")
plt.legend([
    "Ground Truth", 
    "EEG-only Model", 
    "Envelope-only Model", 
    "Fusion (Frozen EEG) Model", 
    "Fusion Model"
])
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("plots/sub001_average_psd_comparison.png", dpi=300)
plt.close()

# Plot a single example PSD
example_idx = 0
plt.figure(figsize=(10, 6))
plt.semilogy(f, psd_ground_truths[example_idx], 'k-', linewidth=2)
plt.semilogy(f, psd_eeg_only[example_idx], 'b-')
plt.semilogy(f, psd_envelope[example_idx], 'g-')
plt.semilogy(f, psd_fusion_frozeneeg[example_idx], 'r-')
plt.semilogy(f, psd_fusion[example_idx], 'm-')

plt.title(f"Example Power Spectral Density for Subject sub-001 (Segment {example_idx})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (log scale)")
plt.legend([
    "Ground Truth", 
    "EEG-only Model", 
    "Envelope-only Model", 
    "Fusion (Frozen EEG) Model", 
    "Fusion Model"
])
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("plots/sub001_example_psd_comparison.png", dpi=300)
plt.close()

# Plot the normalized time series for an example segment
def normalize(x):
    return (x - x.mean()) / x.std()

example_idx = 0
plt.figure(figsize=(12, 6))
plt.plot(normalize(ground_truths[example_idx].reshape(-1)), 'k-', linewidth=2)
plt.plot(normalize(predictions_eeg_only[example_idx].reshape(-1)), 'b-')
plt.plot(normalize(predictions_envelope[example_idx].reshape(-1)), 'g-')
plt.plot(normalize(predictions_fusion_frozeneeg[example_idx].reshape(-1)), 'r-')
plt.plot(normalize(predictions_fusion[example_idx].reshape(-1)), 'm-')

plt.title(f"Example Time Series for Subject sub-001 (Segment {example_idx})")
plt.xlabel("Time (samples)")
plt.ylabel("Normalized Amplitude")
plt.legend([
    "Ground Truth", 
    "EEG-only Model", 
    "Envelope-only Model", 
    "Fusion (Frozen EEG) Model", 
    "Fusion Model"
])
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/sub001_example_timeseries_comparison.png", dpi=300)
plt.close()

# Calculate and print the correlations for sub-001
correlations = []
for i in range(len(ground_truths)):
    gt = ground_truths[i].reshape(-1)
    
    corr_eeg_only = np.corrcoef(gt, predictions_eeg_only[i].reshape(-1))[0, 1]
    corr_envelope = np.corrcoef(gt, predictions_envelope[i].reshape(-1))[0, 1]
    corr_fusion_frozeneeg = np.corrcoef(gt, predictions_fusion_frozeneeg[i].reshape(-1))[0, 1]
    corr_fusion = np.corrcoef(gt, predictions_fusion[i].reshape(-1))[0, 1]
    
    correlations.append({
        "eeg_only": corr_eeg_only,
        "envelope": corr_envelope,
        "fusion_frozeneeg": corr_fusion_frozeneeg,
        "fusion": corr_fusion
    })

# Calculate average correlations
avg_correlations = {
    model: np.mean([corr[model] for corr in correlations]) 
    for model in ["eeg_only", "envelope", "fusion_frozeneeg", "fusion"]
}

print("\nAverage correlations for subject sub-001:")
for model, corr in avg_correlations.items():
    print(f"{model}: {corr:.4f}")

# Plot average correlations
plt.figure(figsize=(8, 6))
model_names = ["EEG-only", "Envelope-only", "Fusion (Frozen EEG)", "Fusion"]
model_keys = ["eeg_only", "envelope", "fusion_frozeneeg", "fusion"]
plt.bar(model_names, [avg_correlations[key] for key in model_keys])
plt.title("Average Correlation with Ground Truth for Subject sub-001")
plt.ylabel("Pearson Correlation")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/sub001_average_correlations.png", dpi=300)
plt.close()

print("\nPlots saved to plots/ directory") 