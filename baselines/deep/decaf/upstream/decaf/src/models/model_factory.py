from typing import Dict, Type, Any
from configs.baseline_config import BaselineConfig

class ModelFactory:
    """Factory class for creating model instances."""
    
    _models: Dict[str, tuple] = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type, config_class: Type[BaselineConfig]):
        """
        Register a model with its configuration class.
        
        Args:
            name (str): Name of the model
            model_class (Type): Model class
            config_class (Type[BaselineConfig]): Configuration class for the model
        """
        cls._models[name] = (model_class, config_class)
    
    @classmethod
    def create(cls, model_name: str, config: BaselineConfig) -> Any:
        """
        Create a model instance.
        
        Args:
            model_name (str): Name of the model to create
            config (BaselineConfig): Configuration for the model
            
        Returns:
            Any: Instance of the requested model
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Creating model: {model_name}")
            
        model_class, _ = cls._models[model_name]
        return model_class(**config.model_params())

    @classmethod
    def list_available_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())

    @classmethod
    def get_config_class(cls, model_name: str) -> Type[BaselineConfig]:
        """
        Get the configuration class for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Type[BaselineConfig]: Configuration class for the model
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        _, config_class = cls._models[model_name]
        return config_class

# Import registry to execute model registrations
from . import registry 