# ===== Model Imports =====
from .model_factory import ModelFactory

# Base Models
from .baseline import BaselineDecoder
from .baseline2 import BaselineDecoder2
from .vlaai import VLAAI

# Quoka Family Models
from .happyquoka import HappyQuoka
from .happyquoka_scaled import HappyQuokaScaled
from .vlaaiquoka import VLAAIQuoka
from .happyquoka_inpaint import HappyQuokaInpaint

# Other Animal Models
from .happycamel import HappyCamel
from .happygoat import HappyGoat
from .happygoat2 import HappyGoat2

# Special Models
from .emma import EMMA
from .envelope_inpaint import EnvelopeInpaint
from .envelope_transformer import EnvelopeTransformer
from .envelope_lstm import EnvelopeLSTM
from .envelope_forecaster import EnvelopeForecaster
from .envelope_forecaster_autoregressive import EnvelopeForecasterAutoregressive
from .happycamel_inpaint import HappyCamelInpaint
from .two_branch_model import TwoBranchModel
from .two_branch_model_recursive import TwoBranchModelRecursive

# ===== Config Imports =====

# Base Configs
from configs.baseline_decoder_config import BaselineDecoderConfig
from configs.baseline_decoder2_config import BaselineDecoder2Config
from configs.vlaai_config import VlaaiConfig

# Quoka Family Configs
from configs.happyquoka_config import HappyQuokaConfig
from configs.happyquoka2_config import HappyQuoka2Config
from configs.happyquoka3_config import HappyQuoka3Config
from configs.happyquoka4_config import HappyQuoka4Config
from configs.vlaaiquoka_config import VLAAIQuokaConfig
from configs.vlaaiquoka2_config import VLAAIQuoka2Config
from configs.happyquoka_inpaint_config import HappyQuokaInpaintConfig

# Camel Family Configs
from configs.happycamel_config import HappyCamelConfig
from configs.happycamel2_config import HappyCamel2Config
from configs.happycamel3_config import HappyCamel3Config
from configs.happycamel4_config import HappyCamel4Config

# Other Animal Configs
from configs.happygoat_config import HappyGoatConfig
from configs.happygoat2_config import HappyGoat2Config

# Special Configs
from configs.emma_config import EMMAConfig
from configs.envelope_inpaint_config import EnvelopeInpaintConfig
from configs.envelope_transformer_config import EnvelopeTransformerConfig
from configs.envelope_lstm_config import EnvelopeLSTMConfig
from configs.envelope_forecaster_config import EnvelopeForecasterConfig
from configs.envelope_forecaster_autoregressive_config import EnvelopeForecasterAutoregressiveConfig
from configs.happycamel_inpaint_config import HappyCamelInpaintConfig

# Two-Branch Model Configs
from configs.two_branch_base_config import TwoBranchBaseConfig
from configs.two_branch_frozen_env_config import TwoBranchFrozenEnvConfig
from configs.two_branch_frozen_eeg_config import TwoBranchFrozenEEGConfig
from configs.two_branch_frozen_both_config import TwoBranchFrozenBothConfig
from configs.two_branch_finetune_env_config import TwoBranchFinetuneEnvConfig
from configs.two_branch_finetune_eeg_config import TwoBranchFinetuneEEGConfig
from configs.two_branch_finetune_both_config import TwoBranchFinetuneBothConfig
from configs.two_branch_recursive_config import TwoBranchRecursiveConfig

# ===== Model Registration =====

# Register Baseline Models
ModelFactory.register_model('baseline_decoder', BaselineDecoder, BaselineDecoderConfig)
ModelFactory.register_model('baseline_decoder2', BaselineDecoder2, BaselineDecoder2Config)
ModelFactory.register_model('vlaai', VLAAI, VlaaiConfig)
ModelFactory.register_model('happyquoka4', HappyQuoka, HappyQuoka4Config)  # Same model class, different config
ModelFactory.register_model('happyquoka_scaled', HappyQuokaScaled, HappyQuoka4Config)
# Register Quoka Family Models
ModelFactory.register_model('happyquoka', HappyQuoka, HappyQuokaConfig)
ModelFactory.register_model('happyquoka2', HappyQuoka, HappyQuoka2Config)  # Same model class, different config
ModelFactory.register_model('happyquoka3', HappyQuoka, HappyQuoka3Config)  # Same model class, different config
ModelFactory.register_model('vlaaiquoka', VLAAIQuoka, VLAAIQuokaConfig)
ModelFactory.register_model('vlaaiquoka2', VLAAIQuoka, VLAAIQuoka2Config)  # Same model class, different learning rate

# Register Other Animal Models
ModelFactory.register_model('happycamel', HappyCamel, HappyCamelConfig)
ModelFactory.register_model('happycamel2', HappyCamel, HappyCamel2Config)  # Register optimized HappyCamel model
ModelFactory.register_model('happycamel3', HappyCamel, HappyCamel3Config)  # Register optimized HappyCamel model
ModelFactory.register_model('happycamel4', HappyCamel, HappyCamel4Config)  # Register optimized HappyCamel model
ModelFactory.register_model('happygoat', HappyGoat, HappyGoatConfig)
ModelFactory.register_model('happygoat2', HappyGoat2, HappyGoat2Config)

# Register Special Models
ModelFactory.register_model('emma', EMMA, EMMAConfig)  # Register EMMA model with multi-rate processing

# ===== Available Model Variants =====

# HappyQuokaInpaint models:
#- happyquoka_inpaint_deterministic_context_1sec
#- happyquoka_inpaint_deterministic_context_2sec
#- happyquoka_inpaint_random_masking_1_9

# EnvelopeInpaint models:
#- envelope_inpaint_deterministic_context_1sec
#- envelope_inpaint_deterministic_context_2sec
#- envelope_inpaint_random_masking_1_9

# TwoBranch models:
#- two_branch_deterministic_context_1sec_weighted
#- two_branch_deterministic_context_2sec_gated
#- two_branch_random_masking_1_9_weighted

# ===== Helper Functions =====

def register_fusion_model(model_class, config_class, model_prefix: str):
    """Register all variants of a fusion model with different strategies and context durations.
    
    Args:
        model_class: The model class to register
        config_class: The configuration class for the model
        model_prefix: Base prefix for the model name (e.g., 'two_branch')
    """
    for context_sec in config_class.CONTEXT_SECONDS:
        for strategy in config_class.FUSION_STRATEGIES:
            # Create a factory function that will set the correct context_seconds
            def create_config(context_sec=context_sec, strategy=strategy):
                return config_class(fusion_strategy=strategy, 
                                    context_seconds=context_sec, 
                                    experiment_name=f"{model_prefix}_context_{float(context_sec)}sec_{strategy}",
                                    )
            
            ModelFactory.register_model(f"{model_prefix}_context_{float(context_sec)}sec_{strategy}", model_class, create_config)
            print(f"Registered model: {model_prefix}_context_{float(context_sec)}sec_{strategy}")
    
def register_inpaint_model(model_class, config_class, model_prefix: str):
    """Register all variants of an inpainting model with different context durations and strategies.
    
    Args:
        model_class: The model class to register
        config_class: The configuration class for the model
        model_prefix: Base prefix for the model name (e.g., 'happyquoka_inpaint')
    """
    # Register deterministic strategy models with different context durations
    for context_sec in config_class.CONTEXT_SECONDS:
        def create_config(context_sec=context_sec):
            return config_class(
                context_mask_strategy='deterministic',
                context_seconds=context_sec,
                experiment_name=f"{model_prefix}_deterministic_context_{int(context_sec)}sec",
            )
        
        ModelFactory.register_model(f"{model_prefix}_deterministic_context_{int(context_sec)}sec", model_class, create_config)
        print(f"Registered model: {model_prefix}_deterministic_context_{int(context_sec)}sec")
    
    def create_random_config():
        return config_class(
            context_mask_strategy='random',
            context_seconds=1.0,  # This value is ignored for random strategy
            experiment_name=f"{model_prefix}_random_masking_1_9",
        )
    
    ModelFactory.register_model(f"{model_prefix}_random_masking_1_9", model_class, create_random_config)
    print(f"Registered model: {model_prefix}_random_masking_1_9")

# ===== Register Advanced Models =====

# Register two-branch models with different configurations
print("\nRegistering two-branch models with different configurations:")

# Both branches trained from scratch
register_fusion_model(TwoBranchModel, TwoBranchBaseConfig, "two_branch")

# Single frozen branch configurations
register_fusion_model(TwoBranchModel, TwoBranchFrozenEnvConfig, "two_branch_frozen_env")
register_fusion_model(TwoBranchModel, TwoBranchFrozenEEGConfig, "two_branch_frozen_eeg")

# Both branches frozen
register_fusion_model(TwoBranchModel, TwoBranchFrozenBothConfig, "two_branch_frozen_both")

# Single fine-tuned branch configurations
register_fusion_model(TwoBranchModel, TwoBranchFinetuneEnvConfig, "two_branch_ft_env")
register_fusion_model(TwoBranchModel, TwoBranchFinetuneEEGConfig, "two_branch_ft_eeg")

# Both branches fine-tuned
register_fusion_model(TwoBranchModel, TwoBranchFinetuneBothConfig, "two_branch_ft_both")

# Recursive model trained from scratch
register_fusion_model(TwoBranchModelRecursive, TwoBranchRecursiveConfig, "two_branch_recursive")

# Register HappyQuokaInpaint models with different context durations and strategies
print("\nRegistering HappyQuokaInpaint models with different context durations and strategies:")
register_inpaint_model(HappyQuokaInpaint, HappyQuokaInpaintConfig, "happyquoka_inpaint")

# Register EnvelopeInpaint models with different context durations and strategies
print("\nRegistering EnvelopeInpaint models with different context durations and strategies:")
register_inpaint_model(EnvelopeInpaint, EnvelopeInpaintConfig, "envelope_inpaint")

# Register EnvelopeTransformer models with different context durations and strategies
print("\nRegistering EnvelopeTransformer models with different context durations and strategies:")
register_inpaint_model(EnvelopeTransformer, EnvelopeTransformerConfig, "envelope_transformer")

# Register EnvelopeLSTM models with different context durations and strategies
print("\nRegistering EnvelopeLSTM models with different context durations and strategies:")
register_inpaint_model(EnvelopeLSTM, EnvelopeLSTMConfig, "envelope_lstm")

# Register HappyCamelInpaint models with different context durations and strategies
print("\nRegistering HappyCamelInpaint models with different context durations and strategies:")
register_inpaint_model(HappyCamelInpaint, HappyCamelInpaintConfig, "happycamel_inpaint")

def register_forecaster_model(model_class, config_class, model_prefix: str):
    """Register forecaster model with different model types.
    
    Args:
        model_class: The model class to register
        config_class: The configuration class for the model
        model_prefix: Base prefix for the model name (e.g., 'envelope_forecaster')
    """
    # Register LSTM forecaster
    def create_lstm_config():
        return config_class(
            rnn_type='lstm',
            experiment_name=f"{model_prefix}_lstm",
        )
    
    ModelFactory.register_model(f"{model_prefix}_lstm", model_class, create_lstm_config)
    print(f"Registered model: {model_prefix}_lstm")
    
    # Register GRU forecaster
    def create_gru_config():
        return config_class(
            rnn_type='gru',
            experiment_name=f"{model_prefix}_gru",
        )
    
    ModelFactory.register_model(f"{model_prefix}_gru", model_class, create_gru_config)
    print(f"Registered model: {model_prefix}_gru")
    
    # Register Transformer forecaster
    def create_transformer_config():
        return config_class(
            rnn_type='transformer',
            n_layers=3,  # Transformers typically benefit from more layers
            hidden_dim=128,
            nhead=8,  # More heads for better attention
            dropout=0.1,
            needs_transformer_schedule=True,
            experiment_name=f"{model_prefix}_transformer",
        )
    
    ModelFactory.register_model(f"{model_prefix}_transformer", model_class, create_transformer_config)
    print(f"Registered model: {model_prefix}_transformer")

# Register EnvelopeForecaster models with different RNN types
print("\nRegistering EnvelopeForecaster models with different RNN types:")
register_forecaster_model(EnvelopeForecaster, EnvelopeForecasterConfig, "envelope_forecaster")

# Register EnvelopeForecasterAutoregressive models with different RNN types
print("\nRegistering EnvelopeForecasterAutoregressive models with different RNN types:")
register_forecaster_model(EnvelopeForecasterAutoregressive, EnvelopeForecasterAutoregressiveConfig, "envelope_forecaster_autoregressive")