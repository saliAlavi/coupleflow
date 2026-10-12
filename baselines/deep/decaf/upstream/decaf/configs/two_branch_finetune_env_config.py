from dataclasses import dataclass
from .two_branch_base_config import TwoBranchBaseConfig

@dataclass
class TwoBranchFinetuneEnvConfig(TwoBranchBaseConfig):
    """Configuration for two-branch model with fine-tuned envelope branch.
    The envelope branch is initialized with pretrained weights and fine-tuned during training.
    The EEG branch is trained from scratch.
    """
    
    # Fine-tuning specific parameters
    env_lr_factor: float = 0.1  # Learning rate factor for pretrained envelope branch
    
    def __post_init__(self):
        """Set experiment name to reflect fine-tuned envelope configuration."""
        super().__post_init__()
        
        # Update experiment name to reflect fine-tuned envelope configuration
        context_str = f"{int(self.context_seconds)}sec"
        self.experiment_name = f"two_branch_finetune_env_{context_str}_{self.fusion_strategy}"
        self.model_name = f"two_branch_finetune_env_{context_str}"
        
    def model_params(self):
        """Return parameters needed for model initialization."""
        params = super().model_params()
        params.update({
            'freeze_env_branch': False,
            'pretrained_env_path': 'checkpoints_forecaster/EnvelopeForecasterLstm_mse/subject_independent/envelope_forecaster_lstm_2layers_128dim_20250225_200010/best_model.pt'
        })
        return params 