import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Literal
from .envelope_inpaint import EnvelopeInpaint
from .happyquoka import HappyQuoka
import matplotlib.pyplot as plt


class TwoBranchModel(nn.Module):
    """
    Flexible two-branch model that combines EEG and envelope processing.
    Can be configured for different scenarios through initialization parameters:
    
    1. Training from scratch:
        - Don't provide pretrained paths
        - Set freeze_*_branch = False
        - Use default learning rate factors
    
    2. Frozen branch(es):
        - Provide pretrained path(s)
        - Set freeze_*_branch = True for branches to freeze
        - Learning rate factors don't matter for frozen branches
    
    3. Fine-tuning branch(es):
        - Provide pretrained path(s)
        - Set freeze_*_branch = False
        - Set appropriate learning rate factors (typically < 1.0)
    
    Examples:
        # Train both from scratch
        model = TwoBranchModel(...)  # Use defaults
        
        # Frozen EEG, train envelope
        model = TwoBranchModel(..., 
                              pretrained_eeg_path='path/to/eeg.pth',
                              freeze_eeg_branch=True)
        
        # Fine-tune both
        model = TwoBranchModel(...,
                              pretrained_eeg_path='path/to/eeg.pth',
                              pretrained_env_path='path/to/env.pth',
                              eeg_lr_factor=0.1,
                              env_lr_factor=0.1)
    """
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        n_layers: int,
        fft_conv1d_kernel: list,
        fft_conv1d_padding: list,
        dropout: float,
        fusion_strategy: Literal['average', 'weighted', 'learned', 'gated'] = 'learned',
        freeze_eeg_branch: bool = False,
        freeze_env_branch: bool = False,
        pretrained_eeg_path: Optional[str] = None,
        pretrained_env_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize envelope branch
        self.envelope_branch = EnvelopeInpaint(
            in_channel=in_channel,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            n_layers=n_layers,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
        )
        
        # Load pretrained weights for envelope branch if provided
        if pretrained_env_path:
            if not os.path.exists(pretrained_env_path):
                raise FileNotFoundError(f"Envelope branch checkpoint not found at {pretrained_env_path}")
            
            envelope_checkpoint = torch.load(pretrained_env_path, map_location='cpu')
            self.envelope_branch.load_state_dict(envelope_checkpoint['model_state_dict'])
            print(f"Loaded pretrained envelope branch from: {pretrained_env_path}")
            self.pretrained_env_path = pretrained_env_path
        
        # Initialize EEG branch
        self.eeg_branch = HappyQuoka(
            in_channel=in_channel,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            n_layers=n_layers,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            g_con=False
        )
        
        # Load pretrained weights for EEG branch if provided
        if pretrained_eeg_path:
            if not os.path.exists(pretrained_eeg_path):
                raise FileNotFoundError(f"EEG branch checkpoint not found at {pretrained_eeg_path}")
            
            eeg_checkpoint = torch.load(pretrained_eeg_path, map_location='cpu')
            self.eeg_branch.load_state_dict(eeg_checkpoint['model_state_dict'])
            print(f"Loaded pretrained EEG branch from: {pretrained_eeg_path}")
            self.pretrained_eeg_path = pretrained_eeg_path
        
        # Freeze branches if specified
        if freeze_eeg_branch:
            for param in self.eeg_branch.parameters():
                param.requires_grad = False
            print("EEG branch frozen")
        else:
            print("EEG branch trainable")
            
        if freeze_env_branch:
            for param in self.envelope_branch.parameters():
                param.requires_grad = False
            print("Envelope branch frozen")
        else:
            print("Envelope branch trainable")
        
        self.fusion_strategy = fusion_strategy
        
        # Initialize fusion-specific parameters
        if fusion_strategy == 'weighted':
            self.weights = nn.Parameter(torch.tensor([0.0, 0.0]))
            
        elif fusion_strategy == 'learned':
            self.fusion_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )
            
        elif fusion_strategy == 'gated':
            self.gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
    
    def get_param_groups(self, base_lr: float) -> list:
        """
        Get parameter groups with their respective learning rates.
        This is used by the optimizer to apply different learning rates to different parts of the model.
        
        Args:
            base_lr: Base learning rate from config
            
        Returns:
            List of dictionaries containing parameters and their learning rates
        """
        param_groups = []
        
        finetune_lr = self.config.get('finetune_lr', base_lr * 0.1)
        
        # EEG branch parameters
        eeg_params = [p for p in self.eeg_branch.parameters() if p.requires_grad]
        if eeg_params:
            eeg_lr = finetune_lr if hasattr(self, 'pretrained_eeg_path') else base_lr
            param_groups.append({
                'params': eeg_params,
                'lr': eeg_lr,
                'is_pretrained': hasattr(self, 'pretrained_eeg_path')
            })
        
        # Envelope branch parameters
        env_params = [p for p in self.envelope_branch.parameters() if p.requires_grad]
        if env_params:
            env_lr = finetune_lr if hasattr(self, 'pretrained_env_path') else base_lr
            param_groups.append({
                'params': env_params,
                'lr': env_lr,
                'is_pretrained': hasattr(self, 'pretrained_env_path')
            })
        
        # Fusion layer parameters
        fusion_params = []
        if self.fusion_strategy == 'weighted':
            fusion_params.extend([self.weights])
        elif self.fusion_strategy in ['learned', 'gated']:
            fusion_params.extend(self.fusion_net.parameters() if self.fusion_strategy == 'learned' 
                               else self.gate_net.parameters())
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': base_lr,
                'is_pretrained': False
            })
        
        # Print learning rate info
        print("\nLearning rate configuration:")
        if eeg_params:
            print(f"EEG branch: {eeg_lr:.2e} ({'pretrained' if hasattr(self, 'pretrained_eeg_path') else 'from scratch'})")
        if env_params:
            print(f"Envelope branch: {env_lr:.2e} ({'pretrained' if hasattr(self, 'pretrained_env_path') else 'from scratch'})")
        if fusion_params:
            print(f"Fusion layer: {base_lr:.2e}")
        
        return param_groups
    
    def fuse_predictions(
        self,
        envelope_pred: torch.Tensor,
        eeg_pred: torch.Tensor,
        envelope_features: Optional[torch.Tensor] = None,
        eeg_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse predictions from both branches based on the selected strategy.
        
        Args:
            envelope_pred: Predictions from envelope branch (batch, seq_len, 1)
            eeg_pred: Predictions from EEG branch (batch, seq_len, 1)
            envelope_features: Optional features from envelope branch for learned fusion
            eeg_features: Optional features from EEG branch for learned fusion
            
        Returns:
            Fused predictions (batch, seq_len, 1)
        """
        if self.fusion_strategy == 'average':
            return (envelope_pred + eeg_pred) / 2
            
        elif self.fusion_strategy == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return weights[0] * envelope_pred + weights[1] * eeg_pred
            
        elif self.fusion_strategy == 'learned':
            if envelope_features is None or eeg_features is None:
                raise ValueError("Features required for learned fusion strategy")
                
            combined_features = torch.cat([envelope_features, eeg_features], dim=-1)
            weights = self.fusion_net(combined_features)
            
            weighted_sum = (
                weights[..., 0:1] * envelope_pred +
                weights[..., 1:2] * eeg_pred
            )
            return weighted_sum
            
        elif self.fusion_strategy == 'gated':
            if envelope_features is None or eeg_features is None:
                raise ValueError("Features required for gated fusion strategy")
                
            combined_features = torch.cat([envelope_features, eeg_features], dim=-1)
            gate = self.gate_net(combined_features)
            
            return gate * envelope_pred + (1 - gate) * eeg_pred
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward(
        self,
        eeg: torch.Tensor,
        envelope_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len, channels)
            envelope_context: Input envelope of shape (batch, seq_len, 1)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope
                'envelope_branch': Envelope branch predictions
                'eeg_branch': EEG branch predictions
        """
        batch_size, seq_len, _ = eeg.shape
        device = eeg.device
        
        envelope_branch_outputs = self.envelope_branch(None, envelope_context)
        eeg_branch_outputs = self.eeg_branch(eeg)
        
        envelope_pred = envelope_branch_outputs['envelope']
        eeg_pred = eeg_branch_outputs['envelope']
        
        eeg_features = None
        envelope_features = None
        if self.fusion_strategy in ['learned', 'gated']:
            if hasattr(self.eeg_branch, 'get_features'):
                eeg_features = self.eeg_branch.get_features()
            if hasattr(self.envelope_branch, 'get_features'):
                envelope_features = self.envelope_branch.get_features()
        
        fused_predictions = self.fuse_predictions(
            envelope_pred=envelope_pred,
            eeg_pred=eeg_pred,
            envelope_features=envelope_features,
            eeg_features=eeg_features
        )
        
        return {
            'envelope': fused_predictions,
            'envelope_branch': envelope_pred,
            'eeg_branch': eeg_pred
        }