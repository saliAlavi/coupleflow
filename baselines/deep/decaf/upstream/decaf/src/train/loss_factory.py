import torch.nn as nn
from typing import Dict, Type, Any
from .losses import PearsonLoss, PearsonL1Loss, MaskedMSELoss, MaskedL1Loss, DistanceCorrelationLoss, DistanceCorrelationL1Loss

class LossFactory:
    """Factory class for creating loss functions."""
    
    _losses: Dict[str, Type[nn.Module]] = {
        "mse": MaskedMSELoss,
        "l1": MaskedL1Loss,
        "pearson": PearsonLoss,
        "pearson_l1": PearsonL1Loss,
        "dcor": DistanceCorrelationLoss,
        "dcor_l1": DistanceCorrelationL1Loss,
    }
    
    @classmethod
    def create(cls, loss_name: str, **kwargs) -> nn.Module:
        """
        Create a loss function instance.
        
        Args:
            loss_name: Name of the loss function to create
            **kwargs: Additional arguments to pass to the loss function
            
        Returns:
            nn.Module: Instance of the requested loss function
        """
        if loss_name not in cls._losses:
            raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {list(cls._losses.keys())}")
            
        loss_class = cls._losses[loss_name]
        return loss_class(**kwargs)
    
    @classmethod
    def list_available_losses(cls) -> list:
        """List all registered loss functions."""
        return list(cls._losses.keys()) 