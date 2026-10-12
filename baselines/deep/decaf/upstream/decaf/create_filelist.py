import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

# Base directory where the data is stored
BASE_DIR = '/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data'

# Default parameters
DEFAULT_WINDOW_LENGTH = 5.0  # in seconds
DEFAULT_SAMPLING_RATE = 64  # Hz
DEFAULT_OVERLAP = 0.8  # 80% overlap

def verify_path(path: str) -> bool:
    """Verify if a file exists and is accessible."""
    return os.path.isfile(path) and os.access(path, os.R_OK)

def get_chunks(length: int, window_length: float = DEFAULT_WINDOW_LENGTH, 
               sampling_rate: int = DEFAULT_SAMPLING_RATE, 
               overlap: float = DEFAULT_OVERLAP) -> List[Tuple[int, int]]:
    """
    Generate start and end indices for chunks with specified overlap.
    
    Args:
        length: Length of the data
        window_length: Length of each window in seconds
        sampling_rate: Sampling rate in Hz
        overlap: Overlap between consecutive windows (0.8 = 80%)
    
    Returns:
        List of tuples containing (start_index, end_index)
    """
    window_size = int(window_length * sampling_rate)
    step = int(window_size * (1 - overlap))
    start_indices = np.arange(0, length - window_size + 1, step)
    end_indices = start_indices + window_size
    return list(zip(start_indices, end_indices))

def get_non_overlapping_chunks(length: int, window_length: float = DEFAULT_WINDOW_LENGTH,
                              sampling_rate: int = DEFAULT_SAMPLING_RATE) -> List[Tuple[int, int]]:
    """
    Generate start and end indices for non-overlapping chunks.
    
    Args:
        length: Length of the data
        window_length: Length of each window in seconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        List of tuples containing (start_index, end_index)
    """
    window_size = int(window_length * sampling_rate)
    start_indices = np.arange(0, length - window_size + 1, window_size)
    end_indices = start_indices + window_size
    return list(zip(start_indices, end_indices))

def parse_file_info(env_path: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse file path to extract subject and audio information.
    
    Args:
        env_path: Path to envelope file
    
    Returns:
        Tuple of (eeg_path, subject, audio_file) or None if parsing fails
    """
    try:
        eeg_path = env_path.replace('envelope.npy', 'eeg.npy')
        parts = os.path.basename(env_path).split('_-_')
        
        if len(parts) != 4:
            print(f"Warning: Invalid filename format: {env_path}")
            return None
            
        return eeg_path, parts[1], parts[2]  # eeg_path, subject, audio_file
    except Exception as e:
        print(f"Error parsing file info from {env_path}: {str(e)}")
        return None

def validate_data_shapes(env_path: str, eeg_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load and validate data shapes from envelope and EEG files.
    
    Args:
        env_path: Path to envelope file
        eeg_path: Path to EEG file
    
    Returns:
        Tuple of (envelope_data, eeg_data) or None if validation fails
    """
    try:
        env_data = np.load(env_path)
        eeg_data = np.load(eeg_path)
        
        # Validate shapes
        if len(env_data.shape) != 2 or env_data.shape[1] != 1:
            print(f"Warning: Invalid envelope shape {env_data.shape} for {env_path}")
            return None
            
        if len(eeg_data.shape) != 2 or eeg_data.shape[1] != 64:
            print(f"Warning: Invalid EEG shape {eeg_data.shape} for {eeg_path}")
            return None
        
        # Validate matching lengths
        if len(env_data) != len(eeg_data):
            print(f"Warning: Length mismatch - Envelope: {len(env_data)}, EEG: {len(eeg_data)} for {env_path}")
            return None
            
        return env_data, eeg_data
        
    except Exception as e:
        print(f"Error loading data from {env_path}: {str(e)}")
        return None

def process_file(env_path: str, base_dir: str, window_length: float = DEFAULT_WINDOW_LENGTH,
                sampling_rate: int = DEFAULT_SAMPLING_RATE, overlap: float = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Process a single envelope file and generate chunk information.
    
    Args:
        env_path: Path to envelope file
        base_dir: Base directory for relative path calculation
        window_length: Length of each window in seconds
        sampling_rate: Sampling rate in Hz
        overlap: Overlap between consecutive windows (0.0 to 1.0)
    
    Returns:
        List of dictionaries containing chunk information
    """
    if not verify_path(env_path):
        print(f"Warning: Cannot access envelope file: {env_path}")
        return []
        
    file_info = parse_file_info(env_path)
    if file_info is None:
        return []
        
    eeg_path, subject, audio_file = file_info
    
    if not verify_path(eeg_path):
        print(f"Warning: Cannot access EEG file: {eeg_path}")
        return []
    
    data_validation = validate_data_shapes(env_path, eeg_path)
    if data_validation is None:
        return []
        
    env_data, _ = data_validation
    
    # Generate chunks based on overlap preference
    if overlap > 0:
        chunks = get_chunks(len(env_data), window_length, sampling_rate, overlap)
    else:
        chunks = get_non_overlapping_chunks(len(env_data), window_length, sampling_rate)
    
    # Store relative paths
    rel_env_path = os.path.relpath(env_path, base_dir)
    rel_eeg_path = os.path.relpath(eeg_path, base_dir)
    
    # Generate chunk information
    return [{
        'sub': subject,
        'audio_file': audio_file,
        'paths_eeg': rel_eeg_path,
        'paths_envelope': rel_env_path,
        'start': start,
        'end': end
    } for start, end in chunks]

def verify_random_paths(df: pd.DataFrame, base_dir: str, num_samples: int = 3) -> None:
    """
    Verify a random sample of paths from the DataFrame.
    
    Args:
        df: DataFrame containing file paths
        base_dir: Base directory for path resolution
        num_samples: Number of random samples to verify
    """
    print("\nVerifying random paths from CSV...")
    sample_rows = df.sample(min(num_samples, len(df)))
    for _, row in sample_rows.iterrows():
        eeg_full_path = os.path.join(base_dir, row['paths_eeg'])
        env_full_path = os.path.join(base_dir, row['paths_envelope'])
        print(f"\nSubject: {row['sub']}, Audio: {row['audio_file']}")
        print(f"EEG path exists: {verify_path(eeg_full_path)}")
        print(f"Envelope path exists: {verify_path(env_full_path)}")

def process_split(split: str, base_dir: str, output_dir: str = 'filelist', 
                 window_length: float = DEFAULT_WINDOW_LENGTH,
                 sampling_rate: int = DEFAULT_SAMPLING_RATE,
                 overlap: float = DEFAULT_OVERLAP) -> Optional[pd.DataFrame]:
    """
    Process a single data split and create corresponding CSV file.
    
    Args:
        split: Split name ('train', 'val', or 'test')
        base_dir: Base directory containing the data
        output_dir: Directory to save the CSV file
        window_length: Length of each window in seconds
        sampling_rate: Sampling rate in Hz
        overlap: Overlap between consecutive windows (0.0 to 1.0)
    
    Returns:
        DataFrame containing processed data or None if processing fails
    """
    overlap_str = f"_{int(overlap*100)}pct" if overlap > 0 else "_no_overlap"
    window_str = f"{int(window_length)}s" if window_length.is_integer() else f"{window_length}s"
    print(f"\nProcessing {split} split (window: {window_str}, overlap: {int(overlap*100)}%)...")
    
    pattern = os.path.join(base_dir, f'{split}_-_*_-_*_-_envelope.npy')
    envelope_files = glob.glob(pattern)
    
    if not envelope_files:
        print(f"Warning: No envelope files found for pattern: {pattern}")
        return None
    
    all_chunks = []
    valid_files = 0
    invalid_files = 0
    
    for env_path in envelope_files:
        chunks = process_file(env_path, base_dir, window_length, sampling_rate, overlap)
        if chunks:
            all_chunks.extend(chunks)
            valid_files += 1
        else:
            invalid_files += 1
    
    if not all_chunks:
        print(f"Warning: No valid data found for {split} split")
        return None
    
    # Create DataFrame and sort
    df = pd.DataFrame(all_chunks)
    df = df.sort_values(['sub', 'audio_file'])
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{split}{overlap_str}_{window_str}.csv')
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print(f"\nSplit: {split}")
    print(f"Window length: {window_length} seconds")
    print(f"Overlap: {int(overlap*100)}%")
    print(f"Valid files processed: {valid_files}")
    print(f"Invalid files skipped: {invalid_files}")
    print(f"Total chunks created: {len(df)}")
    print(f"Number of unique subjects: {df['sub'].nunique()}")
    print(f"Number of unique audio files: {df['audio_file'].nunique()}")
    
    # Verify random paths
    verify_random_paths(df, base_dir)
    
    return df

def main():
    """Main function to process all splits with customizable parameters."""
    parser = argparse.ArgumentParser(description='Create file lists with customizable window length and overlap')
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_LENGTH, 
                        help=f'Window length in seconds (default: {DEFAULT_WINDOW_LENGTH})')
    parser.add_argument('--overlap', type=float, default=DEFAULT_OVERLAP, 
                        help=f'Overlap ratio between 0.0 and 1.0 (default: {DEFAULT_OVERLAP})')
    parser.add_argument('--sampling_rate', type=int, default=DEFAULT_SAMPLING_RATE, 
                        help=f'Sampling rate in Hz (default: {DEFAULT_SAMPLING_RATE})')
    parser.add_argument('--output_dir', type=str, default='filelist', 
                        help='Output directory for CSV files (default: filelist)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], 
                        help='Data splits to process (default: train val test)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process specified splits with custom parameters
    for split in args.splits:
        df = process_split(
            split, 
            BASE_DIR, 
            args.output_dir, 
            window_length=args.window,
            sampling_rate=args.sampling_rate,
            overlap=args.overlap
        )
        if df is None:
            print(f"Failed to process {split} split")
    
    # For backward compatibility, also process test split without overlap if requested
    if 'test' in args.splits or 'DTU' in args.splits and args.overlap > 0:
        no_overlap_dir = f"{args.output_dir}_test_no_overlap"
        os.makedirs(no_overlap_dir, exist_ok=True)
        df = process_split(
            'test', 
            BASE_DIR, 
            no_overlap_dir, 
            window_length=args.window,
            sampling_rate=args.sampling_rate,
            overlap=0.0
        )
        if df is None:
            print("Failed to process test split without overlap")

if __name__ == "__main__":
    main() 