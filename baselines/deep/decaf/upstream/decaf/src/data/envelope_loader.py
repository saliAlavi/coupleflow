import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from scipy import signal

class EnvelopeLoader:
    """Utility class for loading and processing speech envelope data."""
    
    def __init__(
        self,
        base_dir: str,
        filelist_paths: Union[str, List[str]],
        cache_size: int = 100
    ):
        """
        Initialize the envelope loader.
        
        Args:
            base_dir (str): Base directory where the data files are stored
            filelist_paths (Union[str, List[str]]): Path(s) to CSV filelist(s)
            cache_size (int): Number of files to keep in memory cache
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        
        # Convert single path to list
        if isinstance(filelist_paths, str):
            filelist_paths = [filelist_paths]
            
        # Load and combine all filelists
        self.data = []
        for fpath in filelist_paths:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Filelist not found: {fpath}")
            df = pd.read_csv(fpath)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        
        # Initialize cache
        self.cache_size = cache_size
        self._cache: Dict[str, np.ndarray] = {}
        
        # Print dataset info
        print(f"Loaded {len(self.data)} total chunks")
        print(f"Number of unique subjects: {self.data['sub'].nunique()}")
        print(f"Number of unique audio files: {self.data['audio_file'].nunique()}")
    
    def _load_envelope(self, rel_path: str) -> np.ndarray:
        """Load envelope data with caching to avoid repeated disk reads."""
        if rel_path not in self._cache:
            full_path = self.base_dir / rel_path
            try:
                self._cache[rel_path] = np.load(str(full_path))
                # If cache gets too large, remove oldest entries
                if len(self._cache) > self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            except Exception as e:
                raise RuntimeError(f"Error loading {full_path}: {str(e)}")
        return self._cache[rel_path]
    
    def get_envelope_data(self, subject_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load all envelope data, optionally filtered by subject.
        
        Args:
            subject_id (Optional[str]): If provided, load data only for this subject
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping file paths to envelope data arrays
        """
        # Filter by subject if specified
        data = self.data
        if subject_id is not None:
            data = data[data['sub'] == subject_id].reset_index(drop=True)
            if len(data) == 0:
                raise ValueError(f"No data found for subject {subject_id}")
        
        # Load all unique envelope files
        envelopes = {}
        unique_files = data['paths_envelope'].unique()
        
        print(f"Loading {len(unique_files)} unique envelope files...")
        for env_path in unique_files:
            envelopes[env_path] = self._load_envelope(env_path)
            
        return envelopes
    
    def get_chunk_data(self, subject_id: Optional[str] = None) -> List[Dict]:
        """
        Get all chunk information including start/end indices.
        
        Args:
            subject_id (Optional[str]): If provided, get chunks only for this subject
            
        Returns:
            List[Dict]: List of dictionaries containing chunk information
        """
        data = self.data
        if subject_id is not None:
            data = data[data['sub'] == subject_id]
        
        return data.to_dict('records')

if __name__ == "__main__":
    # Example usage
    base_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data"
    filelist_path = "filelist/val.csv"
    
    # Initialize loader
    loader = EnvelopeLoader(base_dir=base_dir, filelist_paths=filelist_path)
    
    # Load all envelopes
    envelopes = loader.get_envelope_data()
    print(f"\nLoaded {len(envelopes)} envelope files")
    
    # Get chunk information
    chunks = loader.get_chunk_data()
    print(f"Total chunks: {len(chunks)}")
    
    # Example: Load envelopes for a specific subject
    subject_envelopes = loader.get_envelope_data(subject_id="sub-001")
    print(f"\nLoaded {len(subject_envelopes)} envelope files for sub-001") 