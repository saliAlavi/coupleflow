import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from scipy import signal

class EEGDataset(Dataset):
    """EEG Dataset for loading EEG and envelope data."""
    
    def __init__(
        self,
        filelist_path: str,
        base_dir: str,
        subject_id: Optional[str] = None,
        transform_eeg: Optional[callable] = None,
        transform_envelope: Optional[callable] = None,
        cache_size: int = 10,
        context_seconds: float = 1.0,
        prediction_seconds: float = 1.0,
        sample_rate: float = 64.0
    ):
        """
        Args:
            filelist_path (str): Path to the CSV filelist
            base_dir (str): Base directory where the data files are stored
            subject_id (Optional[str]): If provided, load data only for this subject
            transform_eeg (callable, optional): Transform to apply to EEG data
            transform_envelope (callable, optional): Transform to apply to envelope data
            cache_size (int): Number of files to keep in memory cache
            context_seconds (float): Number of seconds of the past to consider
            prediction_seconds (float): Number of seconds to predict
            sample_rate (float): Sampling rate of the data in Hz
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")
            
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
            
        self.df = pd.read_csv(filelist_path)
        self.base_dir = base_dir
        self.transform_eeg = transform_eeg
        self.transform_envelope = transform_envelope
        self.cache_size = cache_size
        
        # Context and prediction parameters
        self.context_samples = int(context_seconds * sample_rate)
        self.prediction_samples = int(prediction_seconds * sample_rate)
        self.total_samples = self.context_samples + self.prediction_samples
        self.sample_rate = sample_rate
        self.samples_per_second = int(sample_rate)

        # get the dir of the base dir
        self.raw_audio_dir = os.path.dirname(self.base_dir) + "/preprocessed_stimuli"
        
        # Filter by subject if in subject-dependent mode
        if subject_id is not None:
            self.df = self.df[self.df['sub'] == subject_id].reset_index(drop=True)
            if len(self.df) == 0:
                raise ValueError(f"No data found for subject {subject_id}")
        
        # Cache for loaded data
        self._cache: Dict[str, np.ndarray] = {}
        
        print("Filelist path: ", filelist_path)
        print(f"Dataset initialized with {len(self.df)} chunks")
        print(f"Number of unique subjects: {self.df['sub'].nunique()}")
        print(f"Number of unique audio files: {self.df['audio_file'].nunique()}")
        print(f"Context duration: {self.context_samples/self.sample_rate} seconds")
        print(f"Prediction duration: {self.prediction_samples/self.sample_rate} seconds")
        print(f"Total duration: {self.total_samples/self.sample_rate} seconds")
        print()
    
    def _load_data(self, rel_path: str) -> np.ndarray:
        """Load data with caching to avoid repeated disk reads."""
        if rel_path not in self._cache:
            full_path = os.path.join(self.base_dir, rel_path)
            try:
                self._cache[rel_path] = np.load(full_path)
                # If cache gets too large, remove oldest entries
                if len(self._cache) > self.cache_size:
                    # Remove the first added item (FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            except Exception as e:
                raise RuntimeError(f"Error loading {full_path}: {str(e)}")
        return self._cache[rel_path]

    def _compare_total_samples(self, eeg_data: np.ndarray, envelope_data: np.ndarray) -> bool:
        """Compare the total samples of the EEG and envelope data."""
        return eeg_data.shape[0] == envelope_data.shape[0]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (metadata, eeg_data, envelope_data, envelope_context)
                - metadata: dict containing row information
                - eeg_data: torch.Tensor of shape (sequence_length, 64)
                - envelope_data: torch.Tensor of shape (sequence_length, 1)
                - envelope_context: torch.Tensor of shape (sequence_length, 1)
        """
        # Get row data
        row = self.df.iloc[idx]
        metadata = row.to_dict()
        
        # Load EEG and envelope data
        eeg_data = self._load_data(row['paths_eeg'])
        envelope_data = self._load_data(row['paths_envelope'])

        # Extract the chunk
        start_idx = row['start']
        end_idx = row['end']
        eeg_chunk = eeg_data[start_idx:end_idx]
        envelope_chunk = envelope_data[start_idx:end_idx]

        # Check if the data is of the correct length
        if not self._compare_total_samples(eeg_chunk, envelope_chunk):
            raise ValueError(f"EEG and envelope data have different lengths: {eeg_chunk.shape[0]} != {envelope_chunk.shape[0]} != {self.total_samples}")
        
        # Apply transforms if provided
        if self.transform_eeg:
            eeg_chunk = self.transform_eeg(eeg_chunk)
        if self.transform_envelope:
            envelope_chunk = self.transform_envelope(envelope_chunk)
        
        # Create masked envelope and context mask
        eeg_present = eeg_chunk[-self.prediction_samples:]
        envelope_present = envelope_chunk[-self.prediction_samples:]

        # envelope context
        envelope_context = envelope_chunk[-self.prediction_samples-self.context_samples:-self.prediction_samples]
        eeg_context = eeg_chunk[-self.prediction_samples-self.context_samples:-self.prediction_samples]

        # Convert to torch tensors
        eeg_present = torch.FloatTensor(eeg_present)
        envelope_present = torch.FloatTensor(envelope_present)
        envelope_context = torch.FloatTensor(envelope_context)
        eeg_context = torch.FloatTensor(eeg_context)


        return metadata, eeg_present, envelope_present, eeg_context, envelope_context

def create_dataloader(
    filelist_path: str,
    base_dir: str,
    batch_size: int,
    subject_id: Optional[str] = None,
    transform_eeg: Optional[callable] = None,
    transform_envelope: Optional[callable] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_size: int = 10,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
    context_seconds: float = 2.0,
    prediction_seconds: float = 1.0,
    sample_rate: float = 64.0
) -> DataLoader:
    """Create a DataLoader for the EEG dataset.
    
    Args:
        filelist_path (str): Path to the CSV filelist
        base_dir (str): Base directory where the data files are stored
        batch_size (int): Batch size
        subject_id (Optional[str]): If provided, load data only for this subject
        transform_eeg (callable, optional): Transform to apply to EEG data
        transform_envelope (callable, optional): Transform to apply to envelope data
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        cache_size (int): Number of files to keep in memory cache
        pin_memory (bool): If True, copy tensors into pinned memory (faster GPU transfer)
        prefetch_factor (Optional[int]): Number of samples loaded in advance by each worker
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = EEGDataset_Recursive(
        filelist_path=filelist_path,
        base_dir=base_dir,
        subject_id=subject_id,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        cache_size=cache_size,
        context_seconds=context_seconds,
        prediction_seconds=prediction_seconds,
        sample_rate=sample_rate
    )
    
    # prefetch_factor is only used when num_workers > 0
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    return DataLoader(dataset, **dataloader_kwargs)

# Example usage and transforms
class StandardizeEEG:
    """Standardize EEG data across channels."""
    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        mean = np.mean(eeg_data, axis=0, keepdims=True)
        std = np.std(eeg_data, axis=0, keepdims=True)
        return (eeg_data - mean) / (std + 1e-8)

class MaskingTransform:
    """Apply temporal and/or channel masking to the data."""
    def __init__(
        self,
        time_mask_percent: float = 0.0,
        channel_mask_percent: float = 0.0,
        mask_value: float = 0.0,
        mask_prob: float = 0.5,  # Probability of applying masking
        seed: Optional[int] = None
    ):
        """
        Args:
            time_mask_percent (float): Percentage of time steps to mask (0-1)
            channel_mask_percent (float): Percentage of channels to mask (0-1)
            mask_value (float): Value to use for masking
            mask_prob (float): Probability of applying masking to each sample (0-1)
            seed (Optional[int]): Random seed for reproducibility
        """
        if not (0 <= time_mask_percent <= 1):
            raise ValueError("time_mask_percent must be between 0 and 1")
        if not (0 <= channel_mask_percent <= 1):
            raise ValueError("channel_mask_percent must be between 0 and 1")
        if not (0 <= mask_prob <= 1):
            raise ValueError("mask_prob must be between 0 and 1")
            
        self.time_mask_percent = time_mask_percent
        self.channel_mask_percent = channel_mask_percent
        self.mask_value = mask_value
        self.mask_prob = mask_prob
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
    def _apply_time_mask(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal masking to the data."""
        if self.time_mask_percent == 0:
            return data
            
        seq_len = data.shape[0]
        mask_size = int(seq_len * self.time_mask_percent)
        
        if mask_size > 0:
            # Randomly select start point
            start = self.rng.randint(0, seq_len - mask_size + 1)
            # Create a copy to avoid modifying the original
            data = data.copy()
            # Apply mask
            data[start:start + mask_size] = self.mask_value
            
        return data
    
    def _apply_channel_mask(self, data: np.ndarray) -> np.ndarray:
        """Apply channel masking to the data."""
        if self.channel_mask_percent == 0:
            return data
            
        n_channels = data.shape[1] if len(data.shape) > 1 else 1
        n_mask = int(n_channels * self.channel_mask_percent)
        
        if n_mask > 0 and n_channels > 1:
            # Randomly select channels to mask
            mask_channels = self.rng.choice(n_channels, n_mask, replace=False)
            # Create a copy to avoid modifying the original
            data = data.copy()
            # Apply mask
            data[:, mask_channels] = self.mask_value
            
        return data
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply both temporal and channel masking to the data with probability mask_prob."""
        # Only apply masking with probability mask_prob
        if self.rng.random() < self.mask_prob:
            data = self._apply_time_mask(data)
            data = self._apply_channel_mask(data)
        return data

class EnvelopeFilter:
    """Filter envelope data with configurable filter settings."""
    def __init__(self, filter_type: str = 'none', fs: float = 64.0):
        """
        Args:
            filter_type (str): Type of filter to apply. Options:
                - 'none': No filtering
                - 'lowpass_4hz': 4 Hz lowpass filter
                - 'lowpass_8hz': 8 Hz lowpass filter
                - 'bandpass_1_4hz': 1-4 Hz bandpass filter
                - 'bandpass_4_8hz': 4-8 Hz bandpass filter
                - 'highpass_1hz': 1 Hz highpass filter
            fs (float): Sampling frequency in Hz
        """
        self.filter_type = filter_type
        self.fs = fs
        self.b, self.a = self._design_filter()


        print(f"Filter type: {self.filter_type}, Filter specs: {self.b}, {self.a}")
    
    def _design_filter(self):
        """Design the filter based on specifications."""
        if self.filter_type == 'none':
            return None, None
        
        filter_specs = {
            'lowpass_4hz': (4, 'low'),
            'lowpass_8hz': (8, 'low'),
            'bandpass_1_4hz': ([1, 4], 'band'),
            'bandpass_4_8hz': ([4, 8], 'band'),
            'highpass_1hz': (1, 'high')
        }
        
        if self.filter_type not in filter_specs:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        cutoff, ftype = filter_specs[self.filter_type]
        return signal.butter(4, cutoff, ftype, fs=self.fs)
    
    def __call__(self, envelope_data: np.ndarray) -> np.ndarray:
        """Apply the filter to the envelope data."""
        if self.filter_type == 'none':
            return envelope_data
        
        # Apply filter using filtfilt for zero-phase filtering
        return signal.filtfilt(self.b, self.a, envelope_data, axis=0)

class StandardizeEnvelope:
    """Standardize envelope data with optional filtering."""
    def __init__(self, filter_type: str = 'none', fs: float = 64.0):
        """
        Args:
            filter_type (str): Type of filter to apply before standardization
            fs (float): Sampling frequency in Hz
        """
        self.filter = EnvelopeFilter(filter_type, fs)
    
    def __call__(self, envelope_data: np.ndarray) -> np.ndarray:
        # Apply filter first
        filtered_data = self.filter(envelope_data)
        
        # Then standardize
        # mean = np.mean(filtered_data)
        # std = np.std(filtered_data)
        # return (filtered_data - mean) / (std + 1e-8)

        # Option 2: scale the envelope to be between 0 and 1
        min_val = np.min(filtered_data)
        max_val = np.max(filtered_data)
        return (filtered_data - min_val) / (max_val - min_val)


class EEGDataset_Recursive(Dataset):
    """EEG Dataset for loading EEG and envelope data."""
    
    def __init__(
        self,
        filelist_path: str,
        base_dir: str,
        subject_id: Optional[str] = None,
        transform_eeg: Optional[callable] = None,
        transform_envelope: Optional[callable] = None,
        cache_size: int = 10,
        context_seconds: float = 1.0,
        prediction_seconds: float = 1.0,
        sample_rate: float = 64.0
    ):
        """
        Args:
            filelist_path (str): Path to the CSV filelist
            base_dir (str): Base directory where the data files are stored
            subject_id (Optional[str]): If provided, load data only for this subject
            transform_eeg (callable, optional): Transform to apply to EEG data
            transform_envelope (callable, optional): Transform to apply to envelope data
            cache_size (int): Number of files to keep in memory cache
            context_seconds (float): Number of seconds of the past to consider
            prediction_seconds (float): Number of seconds to predict
            sample_rate (float): Sampling rate of the data in Hz
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")
            
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
            
        self.df = pd.read_csv(filelist_path)
        self.base_dir = base_dir
        self.transform_eeg = transform_eeg
        self.transform_envelope = transform_envelope
        self.cache_size = cache_size
        
        # Context and prediction parameters
        self.context_samples = int(context_seconds * sample_rate)
        self.prediction_samples = int(prediction_seconds * sample_rate)
        self.total_samples = self.context_samples + self.prediction_samples
        self.sample_rate = sample_rate
        self.samples_per_second = int(sample_rate)

        # get the dir of the base dir
        self.raw_audio_dir = os.path.dirname(self.base_dir) + "/preprocessed_stimuli"
        
        # Filter by subject if in subject-dependent mode
        if subject_id is not None:
            self.df = self.df[self.df['sub'] == subject_id].reset_index(drop=True)
            if len(self.df) == 0:
                raise ValueError(f"No data found for subject {subject_id}")
        
        # Cache for loaded data
        self._cache: Dict[str, np.ndarray] = {}
        
        print("Filelist path: ", filelist_path)
        print(f"Dataset initialized with {len(self.df)} chunks")
        print(f"Number of unique subjects: {self.df['sub'].nunique()}")
        print(f"Number of unique audio files: {self.df['audio_file'].nunique()}")
        print(f"Context duration: {self.context_samples/self.sample_rate} seconds")
        print(f"Prediction duration: {self.prediction_samples/self.sample_rate} seconds")
        print(f"Total duration: {self.total_samples/self.sample_rate} seconds")
        print()
    
    def _load_data(self, rel_path: str) -> np.ndarray:
        """Load data with caching to avoid repeated disk reads."""
        if rel_path not in self._cache:
            full_path = os.path.join(self.base_dir, rel_path)
            try:
                self._cache[rel_path] = np.load(full_path)
                # If cache gets too large, remove oldest entries
                if len(self._cache) > self.cache_size:
                    # Remove the first added item (FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            except Exception as e:
                raise RuntimeError(f"Error loading {full_path}: {str(e)}")
        return self._cache[rel_path]

    def _compare_total_samples(self, eeg_data: np.ndarray, envelope_data: np.ndarray) -> bool:
        """Compare the total samples of the EEG and envelope data."""
        return eeg_data.shape[0] == envelope_data.shape[0]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (metadata, eeg_data, envelope_data, envelope_context)
                - metadata: dict containing row information
                - eeg_data: torch.Tensor of shape (sequence_length, 64)
                - envelope_data: torch.Tensor of shape (sequence_length, 1)
                - envelope_context: torch.Tensor of shape (sequence_length, 1)
        """
        # Get row data
        row = self.df.iloc[idx]
        metadata = row.to_dict()
        
        # Load EEG and envelope data
        eeg_data = self._load_data(row['paths_eeg'])
        envelope_data = self._load_data(row['paths_envelope'])

        # Extract the chunk
        start_idx = row['start']
        end_idx = row['end']
        eeg_chunk = eeg_data[start_idx:end_idx]
        envelope_chunk = envelope_data[start_idx:end_idx]

        # get the eeg context
        past_start = row['past_start']
        past_end = row['past_end']
        eeg_context = eeg_data[past_start:past_end]

        # get the envelope context
        envelope_context = envelope_data[past_start:past_end]

        # Check if the data is of the correct length
        if not self._compare_total_samples(eeg_chunk, envelope_chunk):
            raise ValueError(f"EEG and envelope data have different lengths: {eeg_chunk.shape[0]} != {envelope_chunk.shape[0]} != {self.total_samples}")
        
        # Apply transforms if provided
        if self.transform_eeg:
            eeg_chunk = self.transform_eeg(eeg_chunk)
            eeg_context = self.transform_eeg(eeg_context)
        if self.transform_envelope:
            envelope_chunk = self.transform_envelope(envelope_chunk)
            envelope_context = self.transform_envelope(envelope_context)
        
        # Create masked envelope and context mask
        eeg_present = eeg_chunk
        envelope_present = envelope_chunk

        # Convert to torch tensors
        eeg_present = torch.FloatTensor(eeg_present)
        envelope_present = torch.FloatTensor(envelope_present)
        envelope_context = torch.FloatTensor(envelope_context)
        eeg_context = torch.FloatTensor(eeg_context)
        
        #print(f"EEG present shape: {eeg_present.shape}, Envelope present shape: {envelope_present.shape}, Envelope context shape: {envelope_context.shape}, EEG context shape: {eeg_context.shape}")
        return metadata, eeg_present, envelope_present, eeg_context, envelope_context


if __name__ == "__main__":
    # Example usage
    transform_eeg = StandardizeEEG()
    transform_envelope = StandardizeEnvelope()
    
    # Base directory where the data files are stored
    base_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data"
    
    # Subject-independent loader (all subjects)
    loader = create_dataloader(
        filelist_path="filelists/filelist_3s_80pct/train_80pct_3s.csv",
        base_dir=base_dir,
        batch_size=32,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        context_seconds=2.0,
        prediction_seconds=1.0,
        sample_rate=64.0
    )
    
    # Subject-dependent loader (single subject)
    subject_loader = create_dataloader(
        filelist_path="filelists/filelist_3s_80pct/train_80pct_3s.csv",
        base_dir=base_dir,
        batch_size=32,
        subject_id="sub-001",
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        context_seconds=2.0,
        prediction_seconds=1.0,
        sample_rate=64.0
    )
    
    # Example iteration
    for metadata, eegs, envelopes, envelopes_context in loader:
        print(f"Batch shapes: EEG {eegs.shape}, Envelope {envelopes.shape}, Envelope Context {envelopes_context.shape}")
        print(f"Subjects in batch: {metadata['sub']}")
        print(f"Audio files in batch: {metadata['audio_file']}")

        # save the envelope
        import matplotlib.pyplot as plt
        
        # Plot both the context envelope (past) and the present envelope in one plot, starting with the context
        plt.figure(figsize=(10, 4))
        context_data = envelopes_context[0, :, 0].cpu().detach().numpy()
        present_data = envelopes[0, :, 0].cpu().detach().numpy()
        combined_data = np.concatenate((context_data, present_data))
        plt.plot(combined_data, label='Combined Envelope', alpha=0.7)
        plt.plot(range(len(context_data)), context_data, color='red', label='Context Envelope', alpha=0.7)
        plt.title('Combined Context and Present Envelopes')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(f"filelists/filelist_3s_80pct/dataloader_combined_envelope.png")
        plt.close()

        break 