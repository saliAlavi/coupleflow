from dataclasses import dataclass
from .two_branch_base_config import TwoBranchBaseConfig

@dataclass
class TwoBranchFrozenEEGConfig(TwoBranchBaseConfig):
    """Configuration for two-branch model with frozen pretrained EEG branch.
    The EEG branch uses pretrained weights and is frozen during training.
    The envelope branch is trained from scratch.
    """
    needs_transformer_schedule: bool = False
    
    def __post_init__(self):
        """Set experiment name to reflect frozen EEG configuration."""
        #super().__post_init__()
        
        # Update experiment name to reflect frozen EEG configuration
        context_str = f"{int(self.context_seconds)}sec"
        self.experiment_name = f"two_branch_frozen_eeg_{context_str}_{self.fusion_strategy}"
        self.model_name = f"two_branch_frozen_eeg_{context_str}"
        
    def model_params(self):
        """Return parameters needed for model initialization."""
        params = super().model_params()

        if self.loss_function == "pearson":
            params.update({
                'freeze_eeg_branch': True,
                'pretrained_eeg_path': 'checkpoints/Happyquoka4_pearson/subject_independent/happyquoka4_20250302_045700/best_model.pt',
            })
        elif self.loss_function == "pearson_l1":
            params.update({
                'freeze_eeg_branch': True,
                'pretrained_eeg_path': 'checkpoints/Happyquoka4_pearson_l1/subject_independent/happyquoka4_20250311_221720/best_model.pt',
            })
            
        return params