from dataclasses import dataclass
from .two_branch_base_config import TwoBranchBaseConfig

@dataclass
class TwoBranchFinetuneBothConfig(TwoBranchBaseConfig):
    """Configuration for two-branch model with both branches fine-tuned.
    Both EEG and envelope branches are initialized with pretrained weights and fine-tuned during training.
    Uses constant learning rate for fine-tuning.
    """
    
    # Override base learning rate to use fine-tuning rate
    learning_rate: float = 1e-5    # Use fine-tuning learning rate for both branches
    num_epochs: int = 20
    patience: int = 5
    
    def __post_init__(self):
        """Set experiment name to reflect both branches fine-tuned configuration."""
        #super().__post_init__()
        
        # Update experiment name to reflect both branches fine-tuned
        context_str = f"{int(self.context_seconds)}sec"
        self.experiment_name = f"two_branch_finetune_both_{context_str}_{self.fusion_strategy}"
        self.model_name = f"two_branch_finetune_both_{context_str}"
        
    def model_params(self):
        """Return parameters needed for model initialization."""
        params = super().model_params()
        params.update({
            'freeze_eeg_branch': False,
            'freeze_env_branch': False,
            'pretrained_eeg_path': 'checkpoints/Happyquoka4_mse/subject_independent/happyquoka4_20250127_184030/best_model.pt',
            'pretrained_env_path': 'checkpoints_forecaster/EnvelopeForecasterLstm_mse/subject_independent/envelope_forecaster_lstm_2layers_128dim_20250225_200010/best_model.pt'
        })
        return params