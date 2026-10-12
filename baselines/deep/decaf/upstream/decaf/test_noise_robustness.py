import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re
from tqdm import tqdm

# Import functions from visualize_model_predictions.py
from visualize_model_predictions import (
    load_model, normalize_signal, discover_models, load_all_models,
    create_dataloader, StandardizeEEG, StandardizeEnvelope
)

# Define the directory for saving noise test results
NOISE_TEST_DIR = "checkpoints_paper/noise_robustness_results_unified"

def add_noise_to_eeg(eeg_data: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add Gaussian noise to EEG data at a specified SNR level.
    
    Args:
        eeg_data: EEG data tensor of shape (batch_size, seq_len, channels)
        snr_db: Signal-to-Noise Ratio in decibels
        
    Returns:
        Noisy EEG data tensor of the same shape
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate signal power
    signal_power = torch.mean(eeg_data ** 2, dim=(1, 2), keepdim=True)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = torch.randn_like(eeg_data) * torch.sqrt(noise_power)
    
    # Add noise to the signal
    noisy_eeg = eeg_data + noise
    
    return noisy_eeg

def add_noise_to_context(envelope_context: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add Gaussian noise to envelope context data at a specified SNR level.
    
    Args:
        envelope_context: Envelope context data tensor of shape (batch_size, seq_len, channels)
        snr_db: Signal-to-Noise Ratio in decibels
        
    Returns:
        Noisy envelope context data tensor of the same shape
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate signal power
    signal_power = torch.mean(envelope_context ** 2, dim=(1, 2), keepdim=True)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = torch.randn_like(envelope_context) * torch.sqrt(noise_power)
    
    # Add noise to the signal
    noisy_context = envelope_context + noise
    
    return noisy_context

def get_subject_correlations(model, test_batch, device, noise_type=None, snr_db=None, crop_end=None):
    """
    Calculate per-subject correlations for a model on a test batch, optionally with added noise.
    
    Args:
        model: The model to evaluate
        test_batch: Tuple of (metadata, eegs, envelopes, envelope_context)
        device: Device to run the model on
        noise_type: Type of noise to add ('eeg', 'context', or None for no noise)
        snr_db: Signal-to-Noise Ratio in dB for adding noise (None for no noise)
        crop_end: Number of samples to keep from the end for correlation calculation
        
    Returns:
        Dictionary mapping subject IDs to lists of correlation values
    """
    metadata, eegs, envelopes, eeg_context, envelope_context = test_batch
    
    # Add noise if SNR is specified
    if snr_db is not None and noise_type is not None:
        if noise_type == 'eeg':
            eegs = add_noise_to_eeg(eegs, snr_db)
        elif noise_type == 'context':
            envelope_context = add_noise_to_context(envelope_context, snr_db)
    
    # Get model predictions
    with torch.no_grad():
        try:
            outputs = model(eegs.to(device))
        except Exception:
            try:
                outputs = model(eegs.to(device), eeg_context.to(device))
            except Exception as ex:
                try:
                    outputs = model(eegs.to(device), envelope_context.to(device))
                except Exception as ex2:
                    print(f"Error processing model: {ex2}")
                    return {}
    
    # Get predictions from outputs dictionary
    pred = outputs['envelope'].cpu().numpy()
    ground_truth = envelopes.numpy()
    
    # Crop signals if specified
    if crop_end is not None:
        pred = pred[:, :crop_end, :]
        ground_truth = ground_truth[:, :crop_end, :]
    
    # Calculate correlations for each sample
    subject_correlations = {}
    for i in range(len(eegs)):
        subject = metadata['sub'][i]
        
        # Normalize signals
        norm_gt = normalize_signal(ground_truth[i, :, 0])
        norm_pred = normalize_signal(pred[i, :, 0])
        
        # Calculate correlation
        try:
            corr = np.corrcoef(norm_gt, norm_pred)[0, 1]
            # Check if correlation is valid
            if np.isfinite(corr) and not np.isnan(corr):
                if subject not in subject_correlations:
                    subject_correlations[subject] = []
                subject_correlations[subject].append(corr)
        except Exception:
            pass
    
    return subject_correlations

def evaluate_model_with_noise(model, model_name, dataloader, device, noise_type, snr_levels, crop_end=None):
    """
    Evaluate a model's performance across different noise levels.
    
    Args:
        model: The model to evaluate
        model_name: Name of the model
        dataloader: DataLoader for test data
        device: Device to run the model on
        noise_type: Type of noise to add ('eeg', 'context')
        snr_levels: List of SNR levels in dB to test
        crop_end: Number of samples to keep from the end for correlation calculation
        
    Returns:
        DataFrame with performance metrics at each noise level
    """
    results = []
    
    # First evaluate without noise
    print(f"Evaluating {model_name} without noise...")
    clean_subject_correlations = {}
    
    # Process all batches
    for batch in tqdm(dataloader):
        batch_correlations = get_subject_correlations(model, batch, device, noise_type=None, snr_db=None, crop_end=crop_end)
        
        # Merge batch results with overall results
        for subject, corrs in batch_correlations.items():
            if subject not in clean_subject_correlations:
                clean_subject_correlations[subject] = []
            clean_subject_correlations[subject].extend(corrs)
    
    # Calculate mean correlation per subject
    for subject, corrs in clean_subject_correlations.items():
        mean_corr = np.mean(corrs)
        results.append({
            'model': model_name,
            'noise_type': 'None',
            'snr': 'Clean',
            'subject': subject,
            'correlation': mean_corr
        })
    
    # Then evaluate with different noise levels
    for snr_db in snr_levels:
        print(f"Evaluating {model_name} with {noise_type} noise at SNR {snr_db} dB...")
        noisy_subject_correlations = {}
        
        # Process all batches
        for batch in tqdm(dataloader):
            batch_correlations = get_subject_correlations(
                model, batch, device, noise_type=noise_type, snr_db=snr_db, crop_end=crop_end
            )
            
            # Merge batch results with overall results
            for subject, corrs in batch_correlations.items():
                if subject not in noisy_subject_correlations:
                    noisy_subject_correlations[subject] = []
                noisy_subject_correlations[subject].extend(corrs)
        
        # Calculate mean correlation per subject
        for subject, corrs in noisy_subject_correlations.items():
            mean_corr = np.mean(corrs)
            results.append({
                'model': model_name,
                'noise_type': noise_type,
                'snr': f'{snr_db} dB',
                'subject': subject,
                'correlation': mean_corr
            })
    
    return pd.DataFrame(results)

def main(base_dir="/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data",
         test_filelist="filelists/filelist_3s_80pct_test_no_overlap/test_no_overlap_3s.csv",
         snr_levels=None,
         noise_types=None):
    """
    Main function to evaluate models under different noise conditions.
    
    Args:
        base_dir: Base directory for data
        test_filelist: Path to test file list
        snr_levels: List of SNR levels in dB to test (default: [20, 10, 5, 0, -5, -10, -20])
        noise_types: List of noise types to test (default: ['eeg', 'context'])
    """
    if snr_levels is None:
        snr_levels = [10, 5, 0, -5, -10]
    
    if noise_types is None:
        noise_types = ['eeg']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    checkpoint_dir = "checkpoints_paper"
    filtering_type = "none"
    
    # Create output directory
    os.makedirs(NOISE_TEST_DIR, exist_ok=True)
    
    # Set crop length to 1 second (64 samples at 64Hz)
    crop_end = int(1 * 64)
    
    # Define transforms
    transform_eeg = StandardizeEEG()
    transform_envelope = StandardizeEnvelope(
        filter_type=filtering_type,
        fs=64
    )
    
    # Default context and prediction seconds
    default_context_seconds = 1.5
    default_prediction_seconds = 3.5
    
    # Model mappings - combine mappings from both scripts
    model_mappings = {
        "baselinedecoder2": "baseline_decoder2",
        "vlaai": "vlaai",
        "happyquoka2": "happyquoka2",
        "twobranchrecursivecontext3.0secdynamic_pearson_l1": "two_branch_recursive_context_3.0sec_dynamic",
        "twobranchcontext3.0secdynamic": "two_branch_context_3.0sec_dynamic",
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
    
    # Discover and load models
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
        
        model_dataloaders[model_name] = create_dataloader(
            filelist_path=test_filelist,
            base_dir=base_dir,
            batch_size=32,  # Smaller batch size for noise testing
            shuffle=False,
            transform_eeg=transform_eeg,
            transform_envelope=transform_envelope,
            num_workers=0,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds
        )
    
    # Load all models
    models = load_all_models(models_info, device)
    if not models:
        print("No models were successfully loaded. Exiting.")
        return
    
    # Filter models as needed
    allowed_model_names = set(info['name'] for info in models_info)
    models = {name: model for name, model in models.items() if name in allowed_model_names}
    
    # Evaluate models with different noise types and levels
    all_results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating model {model_name}...")
        
        # Get model-specific dataloader
        dataloader = model_dataloaders.get(model_name)
        if dataloader is None:
            print(f"Warning: No dataloader found for model {model_name}. Skipping.")
            continue
        
        # Test each noise type
        for noise_type in noise_types:
            print(f"Testing with {noise_type} noise...")
            
            # Evaluate model with this noise type
            model_results = evaluate_model_with_noise(
                model=model,
                model_name=model_name,
                dataloader=dataloader,
                device=device,
                noise_type=noise_type,
                snr_levels=snr_levels,
                crop_end=crop_end
            )
            
            # Save individual model results
            model_name_clean = model_name.replace('/', '_').replace('\\', '_')
            results_filename = f'{NOISE_TEST_DIR}/noise_results_{noise_type}_{model_name_clean}.csv'
            model_results.to_csv(results_filename, index=False)
            print(f"Saved {noise_type} noise results for {model_name} to {results_filename}")
            
            # Append to all results
            all_results.append(model_results)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_results.to_csv(f'{NOISE_TEST_DIR}/noise_results_all_models.csv', index=False)
        print(f"Saved combined noise results to {NOISE_TEST_DIR}/noise_results_all_models.csv")

if __name__ == "__main__":
    # Run noise performance testing

    base_dir="/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data"
    #test_filelist="filelists/filelist_3s_80pct_test_no_overlap/test_no_overlap_3s.csv"
    test_filelist="filelist_recursive/test_past_present_no_overlap_7.0s_past3.5s_present3.5s.csv"

    main(base_dir, test_filelist)