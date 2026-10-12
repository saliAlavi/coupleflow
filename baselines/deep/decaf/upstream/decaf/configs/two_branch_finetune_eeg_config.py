from dataclasses import dataclass
from .two_branch_base_config import TwoBranchBaseConfig

@dataclass
class TwoBranchFinetuneEEGConfig(TwoBranchBaseConfig):
    """Configuration for two-branch model with fine-tuned EEG branch.
    The EEG branch is initialized with pretrained weights and fine-tuned during training.
    The envelope branch is trained from scratch.
    """
    
    # Fine-tuning specific parameters
    eeg_lr_factor: float = 0.1  # Learning rate factor for pretrained EEG branch
    
    def __post_init__(self):
        """Set experiment name to reflect fine-tuned EEG configuration."""
        #super().__post_init__()
        
        # Update experiment name to reflect fine-tuned EEG configuration
        context_str = f"{int(self.context_seconds)}sec"
        self.experiment_name = f"two_branch_finetune_eeg_{context_str}_{self.fusion_strategy}"
        self.model_name = f"two_branch_finetune_eeg_{context_str}"
        
    def model_params(self):
        """Return parameters needed for model initialization."""
        params = super().model_params()
        params.update({
            'freeze_eeg_branch': False,
            'pretrained_eeg_path': 'checkpoints/Happyquoka4_mse/subject_independent/happyquoka4_20250127_184030/best_model.pt'  # This will be set by train_model.py
        })
        return params 