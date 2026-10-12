from torch.utils.data import DataLoader
from typing import Tuple
from .eeg_dataloader import EEGDataset, StandardizeEEG, StandardizeEnvelope, MaskingTransform, EEGDataset_Recursive
import torch

def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create standardized dataloaders for training, validation, and testing.
    
    Args:
        config: Configuration object containing data parameters
            Required fields:
            - train_filelist: Path to training filelist
            - val_filelist: Path to validation filelist
            - test_filelist: Path to test filelist
            - data_base_dir: Base directory for data files
            - batch_size: Batch size for dataloaders
            - num_workers: Number of workers for dataloaders
            - pin_memory: Whether to use pinned memory
            - prefetch_factor: Number of batches to prefetch per worker
            - subject_id: Optional subject ID for subject-dependent training
            - envelope_filter_type: Type of filter to apply to envelope
            - envelope_sampling_rate: Sampling rate for envelope filtering
            Optional fields for data augmentation:
            - eeg_time_mask_percent: Percentage of time steps to mask in EEG (0-1)
            - eeg_channel_mask_percent: Percentage of channels to mask in EEG (0-1)
            - envelope_time_mask_percent: Percentage of time steps to mask in envelope (0-1)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    def compose_transforms(transforms):
        """Helper function to compose transforms."""
        def composed_transform(x):
            for t in transforms:
                x = t(x)
            return x
        return composed_transform
    
    # Create training transforms (with augmentation)
    train_transform_eeg = [StandardizeEEG()]
    train_transform_envelope = [StandardizeEnvelope(
        filter_type=getattr(config, 'envelope_filter_type', 'none'),
        fs=getattr(config, 'envelope_sampling_rate', 64.0)
    )]
    
    # Add masking transforms for training if specified
    if hasattr(config, 'eeg_time_mask_percent') or hasattr(config, 'eeg_channel_mask_percent'):
        print("\nTraining EEG Masking Configuration:")
        print(f"Time mask percent: {getattr(config, 'eeg_time_mask_percent', 0.0)}")
        print(f"Channel mask percent: {getattr(config, 'eeg_channel_mask_percent', 0.0)}")
        print(f"Mask probability: {getattr(config, 'mask_prob', 0.5)}")
        print()
        train_transform_eeg.append(MaskingTransform(
            time_mask_percent=getattr(config, 'eeg_time_mask_percent', 0.0),
            channel_mask_percent=getattr(config, 'eeg_channel_mask_percent', 0.0),
            mask_value=0.0,
            mask_prob=getattr(config, 'mask_prob', 0.5)  # Default 50% chance of masking
        ))
    
    if hasattr(config, 'envelope_time_mask_percent'):
        print("\nTraining Envelope Masking Configuration:")
        print(f"Time mask percent: {getattr(config, 'envelope_time_mask_percent', 0.0)}")
        print(f"Mask probability: {getattr(config, 'mask_prob', 0.5)}")
        print()
        train_transform_envelope.append(MaskingTransform(
            time_mask_percent=getattr(config, 'envelope_time_mask_percent', 0.0),
            channel_mask_percent=0.0,  # No channel masking for envelope
            mask_value=0.0,
            mask_prob=getattr(config, 'mask_prob', 0.5)  # Default 50% chance of masking
        ))
    
    # Create evaluation transforms (no augmentation)
    eval_transform_eeg = [StandardizeEEG()]
    eval_transform_envelope = [StandardizeEnvelope(
        filter_type=getattr(config, 'envelope_filter_type', 'none'),
        fs=getattr(config, 'envelope_sampling_rate', 64.0)
    )]
    
    # Compose transforms
    train_transform_eeg = compose_transforms(train_transform_eeg)
    train_transform_envelope = compose_transforms(train_transform_envelope)
    eval_transform_eeg = compose_transforms(eval_transform_eeg)
    eval_transform_envelope = compose_transforms(eval_transform_envelope)

    # context mask strategy
    context_seconds = getattr(config, 'context_seconds', 1.0)
    prediction_seconds = getattr(config, 'prediction_seconds', 1.0)

    # Enable cuDNN benchmark mode for optimized performance
    torch.backends.cudnn.benchmark = True

    print(config.is_recursive, "##########################")

    if config.is_recursive:
        # Create datasets with appropriate transforms
        train_dataset = EEGDataset_Recursive(
            filelist_path=config.train_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=train_transform_eeg,
            transform_envelope=train_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
        # mask strategy is always deterministic for validation and test. If random in config it will use context_seconds default in dataloader as start time
        val_dataset = EEGDataset_Recursive(
            filelist_path=config.val_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=eval_transform_eeg,
            transform_envelope=eval_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
        test_dataset = EEGDataset_Recursive(
            filelist_path=config.test_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=eval_transform_eeg,
            transform_envelope=eval_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
    else:
        # Create datasets with appropriate transforms
        train_dataset = EEGDataset(
            filelist_path=config.train_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=train_transform_eeg,
            transform_envelope=train_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
        # mask strategy is always deterministic for validation and test. If random in config it will use context_seconds default in dataloader as start time
        val_dataset = EEGDataset(
            filelist_path=config.val_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=eval_transform_eeg,
            transform_envelope=eval_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
        test_dataset = EEGDataset(
            filelist_path=config.test_filelist,
            base_dir=config.data_base_dir,
            subject_id=config.subject_id,
            transform_eeg=eval_transform_eeg,
            transform_envelope=eval_transform_envelope,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', True),
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        drop_last=True  # Drop last incomplete batch for consistent sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', True),
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', True),
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
    )
    
    print("\nDataset sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\nDataloader settings:")
    print(f"Number of workers: {config.num_workers}")
    print(f"Pin memory: {getattr(config, 'pin_memory', True)}")
    print(f"Prefetch factor: {getattr(config, 'prefetch_factor', 2)}")
    print(f"Batch size: {config.batch_size}")
    
    return train_loader, val_loader, test_loader

def get_recursive_dataloader(config) -> DataLoader:
    """
    Create a test dataloader for recursive models or specialized testing.
    
    Args:
        config: Configuration object containing data parameters
            Required fields:
            - test_filelist: Path to test filelist
            - data_base_dir: Base directory for data files
            - batch_size: Batch size for dataloader
            - num_workers: Number of workers for dataloader
            - pin_memory: Whether to use pinned memory
            - prefetch_factor: Number of batches to prefetch per worker
            - subject_id: Optional subject ID for subject-dependent testing
            - envelope_filter_type: Type of filter to apply to envelope
            - envelope_sampling_rate: Sampling rate for envelope filtering
    
    Returns:
        DataLoader: Test dataloader
    """
    def compose_transforms(transforms):
        """Helper function to compose transforms."""
        def composed_transform(x):
            for t in transforms:
                x = t(x)
            return x
        return composed_transform
    
    # Create evaluation transforms (no augmentation for testing)
    eval_transform_eeg = [StandardizeEEG()]
    eval_transform_envelope = [StandardizeEnvelope(
        filter_type=getattr(config, 'envelope_filter_type', 'none'),
        fs=getattr(config, 'envelope_sampling_rate', 64.0)
    )]
    
    # Compose transforms
    eval_transform_eeg = compose_transforms(eval_transform_eeg)
    eval_transform_envelope = compose_transforms(eval_transform_envelope)

    # Context and prediction settings
    context_seconds = getattr(config, 'context_seconds', 1.0)
    prediction_seconds = getattr(config, 'prediction_seconds', 1.0)

    # Enable cuDNN benchmark mode for optimized performance
    torch.backends.cudnn.benchmark = True
    

    print("Only using the recursive test filelist, filelist_recursive/test_past_present_7.0s_past3.5s_present3.5s.csv")
    
    # Create test dataset
    test_dataset = EEGDataset_Recursive(
        filelist_path="filelist_recursive/test_past_present_no_overlap_7.0s_past3.5s_present3.5s.csv",
        base_dir=config.data_base_dir,
        subject_id=config.subject_id,
        transform_eeg=eval_transform_eeg,
        transform_envelope=eval_transform_envelope,
        context_seconds=context_seconds,
        prediction_seconds=prediction_seconds,
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', True),
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
    )
    
    print(f"\nTest dataset size: {len(test_dataset)}")
    print("\nTest dataloader settings:")
    print(f"Number of workers: {config.num_workers}")
    print(f"Pin memory: {getattr(config, 'pin_memory', True)}")
    print(f"Prefetch factor: {getattr(config, 'prefetch_factor', 2)}")
    print(f"Batch size: {config.batch_size}")
    
    return test_loader 