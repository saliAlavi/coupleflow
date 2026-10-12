from typing import Dict, Any, Optional, Union
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)

class OptimizerScheduler:
    """Class to handle different learning rate scheduling strategies."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_type: str,
        config: Dict[str, Any],
        num_epochs: Optional[int] = None,
        num_training_steps: Optional[int] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_type: Type of scheduler to use. Options:
                - 'step': Step decay learning rate
                - 'cosine': Cosine annealing
                - 'one_cycle': One cycle policy
                - 'plateau': Reduce on plateau
                - 'transformer': Transformer-style warmup + decay
                - None: No scheduling (constant learning rate)
            config: Configuration dictionary containing scheduler parameters
            num_epochs: Total number of epochs (required for some schedulers)
            num_training_steps: Total number of training steps (required for some schedulers)
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.config = config
        self.num_epochs = num_epochs
        self.num_training_steps = num_training_steps
        
        # Get base learning rate
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Initialize scheduler based on type
        self.scheduler = self._initialize_scheduler()
        
        # For transformer schedule
        if scheduler_type == 'transformer':
            self.d_model = config.get('d_model', 128)  # Model dimension
            
            # Calculate warmup steps from epochs
            warmup_epochs = config.get('warmup_epochs', 5)  # Get warmup epochs from config
            if num_training_steps and num_epochs:
                steps_per_epoch = num_training_steps // num_epochs
                self.n_warmup_steps = warmup_epochs * steps_per_epoch
            else:
                self.n_warmup_steps = 40000  # Fallback default
            
            self.n_current_steps = 0
            self.init_lr = np.power(self.d_model, -0.5)  # Initial learning rate based on model dimension
            
        self.frozen_lr = None  # For frozen learning rate support
    
    def _initialize_scheduler(self) -> Optional[Union[_LRScheduler, ReduceLROnPlateau]]:
        """Initialize the appropriate scheduler based on type."""
        if self.scheduler_type is None:
            return None
            
        if self.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 30),
                gamma=self.config.get('scheduler_gamma', 0.1)
            )
            
        elif self.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs if self.num_epochs else 100,
                eta_min=self.config.get('min_lr', 0)
            )
            
        elif self.scheduler_type == 'one_cycle':
            if not self.num_training_steps:
                raise ValueError("num_training_steps required for one_cycle scheduler")
            return OneCycleLR(
                self.optimizer,
                max_lr=self.base_lr,
                total_steps=self.num_training_steps,
                pct_start=self.config.get('warmup_ratio', 0.3),
                div_factor=self.config.get('div_factor', 25.0),
                final_div_factor=self.config.get('final_div_factor', 1e4)
            )
            
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('scheduler_gamma', 0.1),
                patience=self.config.get('scheduler_patience', 10),
                min_lr=self.config.get('min_lr', 1e-6)
            )
            
        elif self.scheduler_type == 'transformer':
            # For transformer, we'll handle the scheduling manually
            return None
            
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def _transformer_schedule_step(self) -> None:
        """Implement Noam scheduler (transformer paper's original implementation)."""
        self.n_current_steps += 1
        lr_reduction_factor = 1  # reduce overall LR by 10x
        
        # Calculate learning rate scale
        lr_scale = np.min([
            np.power(self.n_current_steps, -0.5),  # Decay term
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps  # Warmup term
        ])
        
        # Calculate learning rate
        lr = self.init_lr * lr_scale * lr_reduction_factor
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None) -> None:
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch number (used by some schedulers)
            metrics: Validation metrics (used by ReduceLROnPlateau)
        """
        if self.frozen_lr is not None:
            self._update_frozen_lr()
            return
            
        if self.scheduler_type is None:
            return
            
        if self.scheduler_type == 'transformer':
            self._transformer_schedule_step()
        elif self.scheduler_type == 'plateau':
            if metrics is None:
                raise ValueError("metrics required for plateau scheduler")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def freeze_learning_rate(self, lr: float) -> None:
        """Freeze the learning rate to a specific value."""
        self.frozen_lr = lr
        self._update_frozen_lr()
    
    def unfreeze_learning_rate(self) -> None:
        """Unfreeze the learning rate and resume scheduling."""
        self.frozen_lr = None
    
    def _update_frozen_lr(self) -> None:
        """Update optimizer with frozen learning rate."""
        if self.frozen_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.frozen_lr
    
    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        state = {
            'scheduler_type': self.scheduler_type,
            'base_lr': self.base_lr,
            'frozen_lr': self.frozen_lr
        }
        
        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            state['scheduler_state'] = self.scheduler.state_dict()
            
        if self.scheduler_type == 'transformer':
            state.update({
                'd_model': self.d_model,
                'warmup_steps': self.n_warmup_steps,
                'n_current_steps': self.n_current_steps,
                'init_lr': self.init_lr
            })
            
        return state
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load scheduler state from checkpoint."""
        if state_dict['scheduler_type'] != self.scheduler_type:
            raise ValueError(
                f"Checkpoint scheduler type {state_dict['scheduler_type']} "
                f"does not match current scheduler type {self.scheduler_type}"
            )
            
        self.base_lr = state_dict['base_lr']
        self.frozen_lr = state_dict.get('frozen_lr', None)
        
        if self.scheduler is not None and 'scheduler_state' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
            
        if self.scheduler_type == 'transformer':
            self.d_model = state_dict['d_model']
            self.n_warmup_steps = state_dict['warmup_steps']
            self.n_current_steps = state_dict['n_current_steps']
            self.init_lr = state_dict['init_lr'] 