import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

# Base directory where the data is stored
BASE_DIR = '/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data'

# Default parameters for new chunking logic
DEFAULT_TOTAL_WINDOW_LENGTH = 7.0  # Total window is 7 seconds
DEFAULT_PAST_WINDOW_LENGTH = 3.5   # First 3.5 seconds (past)
DEFAULT_PRESENT_WINDOW_LENGTH = 3.5  # Last 3.5 seconds (present)
DEFAULT_SAMPLING_RATE = 64  # Hz
DEFAULT_OVERLAP = 0.0  # No overlap by default

def verify_path(path: str) -> bool:
    """Verify if a file exists and is accessible."""
    return os.path.isfile(path) and os.access(path, os.R_OK)

def get_past_present_chunks(length: int, 
                           total_window_length: float = DEFAULT_TOTAL_WINDOW_LENGTH,
                           past_window_length: float = DEFAULT_PAST_WINDOW_LENGTH,
                           present_window_length: float = DEFAULT_PRESENT_WINDOW_LENGTH,
                           sampling_rate: int = DEFAULT_SAMPLING_RATE,
                           overlap: float = DEFAULT_OVERLAP) -> List[Tuple[int, int, int, int]]:
    """
    Generate start and end indices for chunks split into past and present, with optional overlap.
    
    Args:
        length: Length of the data
        total_window_length: Total window length in seconds (7.0)
        past_window_length: Past window length in seconds (3.5)
        present_window_length: Present window length in seconds (3.5)
        sampling_rate: Sampling rate in Hz
        overlap: Overlap between consecutive windows (0.0 to 1.0)
    
    Returns:
        List of tuples containing (past_start, past_end, present_start, present_end)
    """
    total_window_size = int(total_window_length * sampling_rate)  # 7s * 64Hz = 448 samples
    past_window_size = int(past_window_length * sampling_rate)    # 3.5s * 64Hz = 224 samples
    present_window_size = int(present_window_length * sampling_rate)  # 3.5s * 64Hz = 224 samples
    
    if overlap > 0:
        # Overlapping windows
        step = int(total_window_size * (1 - overlap))
        start_indices = np.arange(0, length - total_window_size + 1, step)
    else:
        # Non-overlapping windows
        start_indices = np.arange(0, length - total_window_size + 1, total_window_size)
    
    chunks = []
    for start_idx in start_indices:
        # Past: first 3.5 seconds of the 7-second window
        past_start = start_idx
        past_end = start_idx + past_window_size
        
        # Present: last 3.5 seconds of the 7-second window
        present_start = start_idx + past_window_size
        present_end = start_idx + total_window_size
        
        chunks.append((past_start, past_end, present_start, present_end))
    
    return chunks

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

def process_file(env_path: str, base_dir: str, 
                total_window_length: float = DEFAULT_TOTAL_WINDOW_LENGTH,
                past_window_length: float = DEFAULT_PAST_WINDOW_LENGTH,
                present_window_length: float = DEFAULT_PRESENT_WINDOW_LENGTH,
                sampling_rate: int = DEFAULT_SAMPLING_RATE,
                overlap: float = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Process a single envelope file and generate past-present chunk information.
    
    Args:
        env_path: Path to envelope file
        base_dir: Base directory for relative path calculation
        total_window_length: Total window length in seconds (7.0)
        past_window_length: Past window length in seconds (3.5)
        present_window_length: Present window length in seconds (3.5)
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
    
    # Generate past-present chunks
    chunks = get_past_present_chunks(
        len(env_data), 
        total_window_length, 
        past_window_length, 
        present_window_length, 
        sampling_rate,
        overlap
    )
    
    # Store relative paths
    rel_env_path = os.path.relpath(env_path, base_dir)
    rel_eeg_path = os.path.relpath(eeg_path, base_dir)
    
    # Generate chunk information
    return [{
        'sub': subject,
        'audio_file': audio_file,
        'paths_eeg': rel_eeg_path,
        'paths_envelope': rel_env_path,
        'past_start': past_start,
        'past_end': past_end,
        'start': present_start,
        'end': present_end
    } for past_start, past_end, present_start, present_end in chunks]

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
        print(f"Past chunk: {row['past_start']} - {row['past_end']}")
        print(f"Present chunk: {row['start']} - {row['end']}")

def process_split(split: str, base_dir: str, output_dir: str = 'filelist_recursive', 
                  total_window_length: float = DEFAULT_TOTAL_WINDOW_LENGTH,
                  past_window_length: float = DEFAULT_PAST_WINDOW_LENGTH,
                  present_window_length: float = DEFAULT_PRESENT_WINDOW_LENGTH,
                  sampling_rate: int = DEFAULT_SAMPLING_RATE,
                  overlap: float = DEFAULT_OVERLAP) -> Optional[pd.DataFrame]:
    """
    Process a single data split and create corresponding CSV file with past-present chunks.
    
    Args:
        split: Split name ('train', 'val', or 'test')
        base_dir: Base directory containing the data
        output_dir: Directory to save the CSV file
        total_window_length: Total window length in seconds (7.0)
        past_window_length: Past window length in seconds (3.5)
        present_window_length: Present window length in seconds (3.5)
        sampling_rate: Sampling rate in Hz
        overlap: Overlap between consecutive windows (0.0 to 1.0)
    
    Returns:
        DataFrame containing processed data or None if processing fails
    """
    overlap_str = f"_{int(overlap*100)}pct" if overlap > 0 else "_no_overlap"
    window_str = f"{total_window_length}s_past{past_window_length}s_present{present_window_length}s"
    print(f"\nProcessing {split} split with past-present chunks (total: {total_window_length}s, past: {past_window_length}s, present: {present_window_length}s, overlap: {int(overlap*100)}%)...")
    
    pattern = os.path.join(base_dir, f'{split}_-_*_-_*_-_envelope.npy')
    envelope_files = glob.glob(pattern)
    
    if not envelope_files:
        print(f"Warning: No envelope files found for pattern: {pattern}")
        return None
    
    all_chunks = []
    valid_files = 0
    invalid_files = 0
    
    for env_path in envelope_files:
        chunks = process_file(
            env_path, base_dir, 
            total_window_length, 
            past_window_length, 
            present_window_length, 
            sampling_rate,
            overlap
        )
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
    csv_path = os.path.join(output_dir, f'{split}_past_present{overlap_str}_{window_str}.csv')
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print(f"\nSplit: {split}")
    print(f"Total window length: {total_window_length} seconds")
    print(f"Past window length: {past_window_length} seconds")
    print(f"Present window length: {present_window_length} seconds")
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
    """Main function to process specified splits with past-present chunking."""
    parser = argparse.ArgumentParser(description='Create file lists with past-present chunking (7s total: 3.5s past + 3.5s present)')
    parser.add_argument('--total_window', type=float, default=DEFAULT_TOTAL_WINDOW_LENGTH, 
                        help=f'Total window length in seconds (default: {DEFAULT_TOTAL_WINDOW_LENGTH})')
    parser.add_argument('--past_window', type=float, default=DEFAULT_PAST_WINDOW_LENGTH, 
                        help=f'Past window length in seconds (default: {DEFAULT_PAST_WINDOW_LENGTH})')
    parser.add_argument('--present_window', type=float, default=DEFAULT_PRESENT_WINDOW_LENGTH, 
                        help=f'Present window length in seconds (default: {DEFAULT_PRESENT_WINDOW_LENGTH})')
    parser.add_argument('--sampling_rate', type=int, default=DEFAULT_SAMPLING_RATE, 
                        help=f'Sampling rate in Hz (default: {DEFAULT_SAMPLING_RATE})')
    parser.add_argument('--overlap', type=float, default=DEFAULT_OVERLAP, 
                        help=f'Overlap ratio between 0.0 and 1.0 (default: {DEFAULT_OVERLAP})')
    parser.add_argument('--train_overlap', type=float, default=0.8, 
                        help='Overlap ratio for train split specifically (default: 0.8)')
    parser.add_argument('--output_dir', type=str, default='filelist_recursive', 
                        help='Output directory for CSV files (default: filelist_recursive)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], 
                        help='Data splits to process (default: train val test)')
    
    args = parser.parse_args()
    
    # Validate that past + present = total
    if abs(args.past_window + args.present_window - args.total_window) > 0.001:
        print(f"Error: Past window ({args.past_window}s) + Present window ({args.present_window}s) must equal Total window ({args.total_window}s)")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process specified splits with past-present chunking
    for split in args.splits:
        # Use specific overlap for train split, default overlap for others
        split_overlap = args.train_overlap if split == 'train' else args.overlap
        
        df = process_split(
            split,
            BASE_DIR, 
            args.output_dir, 
            total_window_length=args.total_window,
            past_window_length=args.past_window,
            present_window_length=args.present_window,
            sampling_rate=args.sampling_rate,
            overlap=split_overlap
        )
        
        if df is None:
            print(f"Failed to process {split} split")
        else:
            print(f"\n{split.capitalize()} filelist with past-present chunks created successfully!")

if __name__ == "__main__":
    main()