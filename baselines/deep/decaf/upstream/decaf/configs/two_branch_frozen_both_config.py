from dataclasses import dataclass
from .two_branch_base_config import TwoBranchBaseConfig

@dataclass
class TwoBranchFrozenBothConfig(TwoBranchBaseConfig):
    """Configuration for two-branch model with both branches frozen.
    Both EEG and envelope branches use pretrained weights and are frozen during training.
    Only the fusion layer is trained.
    """

    num_epochs: int = 10
    learning_rate: float = 1e-4
    
    def __post_init__(self):
        """Set experiment name to reflect both branches frozen configuration."""
        #super().__post_init__()
        
        # Update experiment name to reflect both branches frozen
        context_str = f"{int(self.context_seconds)}sec"
        self.experiment_name = f"two_branch_frozen_both_{context_str}_{self.fusion_strategy}"
        self.model_name = f"two_branch_frozen_both_{context_str}"
        
    def model_params(self):
        """Return parameters needed for model initialization."""
        params = super().model_params()
        params.update({
            'freeze_eeg_branch': True,
            'freeze_env_branch': True,
            'pretrained_eeg_path': 'checkpoints/Happyquoka4_mse/subject_independent/happyquoka4_20250127_184030/best_model.pt',
            'pretrained_env_path': 'checkpoints_forecaster/EnvelopeForecasterLstm_mse/subject_independent/envelope_forecaster_lstm_2layers_128dim_20250225_200010/best_model.pt'
        })
        return params