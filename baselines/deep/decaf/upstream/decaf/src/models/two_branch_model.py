import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Literal
from .envelope_forecaster import EnvelopeForecaster
from .happyquoka import HappyQuoka
import matplotlib.pyplot as plt


class TwoBranchModel(nn.Module):
    """
    Two-branch model for EEG-to-envelope prediction.
    
    Combines:
    1. Envelope branch: Uses EnvelopeForecaster to predict future envelope from past envelope
    2. EEG branch: Uses HappyQuoka to predict envelope from EEG signals
    
    Supports different fusion strategies:
    - 'average': Simple average of both predictions
    - 'weighted': Weighted average with learnable weights
    - 'dynamic': Dynamic weighting based on input features
    - 'early_f': Early fusion where envelope prediction is fed into EEG branch
    - 'allow_one_branch': Uses a network to select which branch to use instead of combining them
    - 'envelope_only': Uses only the envelope branch, ignoring the EEG branch
    
    Usage examples:
        # From scratch
        model = TwoBranchModel(...)
        
        # Fine-tune EEG branch only
        model = TwoBranchModel(...,
                               pretrained_env_path='path/to/env.pth',
                               freeze_env_branch=True)
        
        # Fine-tune envelope branch only
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
        fusion_strategy: Literal['average', 'weighted', 'dynamic', 'early_f', 'allow_one_branch', 'envelope_only'] = 'average',
        freeze_eeg_branch: bool = False,
        freeze_env_branch: bool = False,
        pretrained_eeg_path: Optional[str] = None,
        pretrained_env_path: Optional[str] = None,
        rnn_type: str = "lstm",  # Options for envelope forecaster: "lstm", "gru", "transformer"
        context_seconds: float = 2.0,
        prediction_seconds: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Initialize envelope branch with EnvelopeForecaster
        self.envelope_branch = EnvelopeForecaster(
            in_channel=1,
            hidden_dim=128,
            n_layers=2,
            dropout=0.1,
            rnn_type=rnn_type,
            nhead=4,
            context_seconds=context_seconds,
            prediction_seconds=prediction_seconds,
        )
        
        # Load pretrained weights for envelope branch if provided
        if pretrained_env_path:
            if not os.path.exists(pretrained_env_path):
                raise FileNotFoundError(f"Envelope branch checkpoint not found at {pretrained_env_path}")
            
            envelope_checkpoint = torch.load(pretrained_env_path, map_location='cuda:0')
            self.envelope_branch.load_state_dict(envelope_checkpoint['model_state_dict'])
            print(f"Loaded pretrained envelope branch from: {pretrained_env_path}")
            self.pretrained_env_path = pretrained_env_path
        
        # For early fusion, we need to modify the input channel count for the EEG branch
        eeg_in_channel = in_channel
        if fusion_strategy == 'early_f':
            # Add 1 channel for the envelope prediction
            eeg_in_channel = in_channel + 1
            # Force envelope branch to be frozen for early fusion
            freeze_env_branch = True
            print("Early fusion strategy selected: Envelope branch will be frozen")
        elif fusion_strategy == 'envelope_only':
            # Force EEG branch to be frozen when using envelope only
            freeze_eeg_branch = True
            print("Envelope-only strategy selected: EEG branch will be frozen and ignored")
        
        # Initialize EEG branch
        self.eeg_branch = HappyQuoka(
            in_channel=eeg_in_channel,
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
            
            eeg_checkpoint = torch.load(pretrained_eeg_path, map_location='cuda:0')
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
            
        if freeze_env_branch or fusion_strategy == 'early_f':
            for param in self.envelope_branch.parameters():
                param.requires_grad = False
            print("Envelope branch frozen")
        else:
            print("Envelope branch trainable")
        
        self.fusion_strategy = fusion_strategy
        
        # Initialize fusion-specific parameters
        if fusion_strategy == 'weighted':
            self.weights = nn.Parameter(torch.tensor([0.2, 0.8]))
        elif fusion_strategy == 'dynamic':
            # Dynamic fusion network that takes both predictions and outputs weights
            # Input: concatenated predictions [envelope_pred, eeg_pred]
            # Output: weight for envelope prediction (1-weight for eeg prediction)
            self.dynamic_fusion_net = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=5, padding=2),  # Temporal convolution to capture local patterns
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, 1, kernel_size=1),
                nn.Sigmoid()  # Output between 0 and 1 for weighting
            )
        elif fusion_strategy == 'allow_one_branch':
            # Branch selector network that takes both predictions and outputs a binary decision
            # Input: concatenated predictions [envelope_pred, eeg_pred]
            # Output: single probability value for selecting envelope branch (0 = eeg branch, 1 = envelope branch)
            self.branch_selector_net = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=5, padding=2),  # Temporal convolution to capture local patterns
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Global average pooling to get a single value per channel
                nn.Flatten(),  # Flatten to (batch, 8)
                nn.Linear(8, 1),  # Final linear layer to get a single value
                nn.Sigmoid()  # Output between 0 and 1 for branch selection
            )

        print(f"TwoBranchModel initialized with:")
        print(f"  - Fusion strategy: {fusion_strategy}")
        print(f"  - Freeze EEG branch: {freeze_eeg_branch}")
        print(f"  - Freeze envelope branch: {freeze_env_branch}")
        print(f"  - Context seconds: {context_seconds}")
        print(f"  - Prediction seconds: {prediction_seconds}")
    
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
        elif self.fusion_strategy == 'dynamic':
            fusion_params.extend(self.dynamic_fusion_net.parameters())
        elif self.fusion_strategy == 'allow_one_branch':
            fusion_params.extend(self.branch_selector_net.parameters())
        
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
        eeg_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse predictions from both branches based on the selected strategy.
        Supports 'average', 'weighted', 'dynamic', 'early_f', 'allow_one_branch', and 'envelope_only' fusion strategies.
        
        Args:
            envelope_pred: Predictions from envelope branch (batch, seq_len, 1)
            eeg_pred: Predictions from EEG branch (batch, seq_len, 1)
            
        Returns:
            Tuple containing:
                - Fused predictions (batch, seq_len, 1)
                - Fusion weights (shape depends on fusion strategy)
        """
        batch_size, seq_len, _ = envelope_pred.shape
        device = envelope_pred.device
        
        if self.fusion_strategy == 'average':
            fused = (envelope_pred + eeg_pred) / 2
            # For average, weights are fixed at [0.5, 0.5]
            weights = torch.tensor([0.5, 0.5], device=device)
            return fused, weights
            
        elif self.fusion_strategy == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            fused = weights[0] * envelope_pred + weights[1] * eeg_pred
            # Return the learned weights
            return fused, weights.detach()
            
        elif self.fusion_strategy == 'dynamic':
            # Stack predictions along channel dimension for the fusion network
            # From (batch, seq_len, 1) to (batch, 2, seq_len)
            stacked_preds = torch.cat([
                envelope_pred.transpose(1, 2),  # (batch, 1, seq_len)
                eeg_pred.transpose(1, 2)        # (batch, 1, seq_len)
            ], dim=1)  # (batch, 2, seq_len)
            
            # Get dynamic weights from fusion network
            # Output shape: (batch, 1, seq_len)
            envelope_weights = self.dynamic_fusion_net(stacked_preds)
            
            # Reshape weights to match prediction shape
            # From (batch, 1, seq_len) to (batch, seq_len, 1)
            envelope_weights = envelope_weights.transpose(1, 2)
            
            # Apply weights to predictions
            # envelope_weights for envelope_pred, (1-envelope_weights) for eeg_pred
            fused = envelope_weights * envelope_pred + (1 - envelope_weights) * eeg_pred
            
            return fused, envelope_weights.detach()
            
        elif self.fusion_strategy == 'early_f':
            # For early fusion, we just return the EEG branch output
            # since fusion happens at the input level
            # For early fusion, weights are implicitly [0, 1] (all weight on EEG branch)
            weights = torch.tensor([0.0, 1.0], device=device)
            return eeg_pred, weights
            
        elif self.fusion_strategy == 'allow_one_branch':
            # Stack predictions along channel dimension for the branch selector network
            # From (batch, seq_len, 1) to (batch, 2, seq_len)
            stacked_preds = torch.cat([
                envelope_pred.transpose(1, 2),  # (batch, 1, seq_len)
                eeg_pred.transpose(1, 2)        # (batch, 1, seq_len)
            ], dim=1)  # (batch, 2, seq_len)
            
            # Get branch selection probability from branch selector network
            # Output shape: (batch, 1)
            branch_probs = self.branch_selector_net(stacked_preds)
            
            # During training, use soft selection with probabilities
            if self.training:
                # Soft selection: weighted combination based on probabilities
                # Expand branch_probs to match the shape of predictions: (batch, 1) -> (batch, seq_len, 1)
                branch_probs_expanded = branch_probs.unsqueeze(1).expand(-1, seq_len, -1)
                fused = branch_probs_expanded * envelope_pred + (1 - branch_probs_expanded) * eeg_pred
            else:
                # Hard selection during inference: binary decision based on probability threshold
                # Create a binary mask: 1 for envelope branch, 0 for EEG branch
                branch_mask = (branch_probs > 0.5).float()
                
                # Select either envelope_pred or eeg_pred for the entire sequence
                # based on the single branch_mask value
                fused = torch.where(
                    branch_mask.unsqueeze(1).expand(-1, seq_len, -1) > 0.5,
                    envelope_pred,
                    eeg_pred
                )
            
            return fused, branch_probs.detach()
            
        elif self.fusion_strategy == 'envelope_only':
            # For envelope-only, we just return the envelope branch output
            # For envelope_only, the weight is implicitly 1.0 for envelope branch
            # the shape of the weights is (B, 1, 1)
            weights = torch.ones(batch_size, 1, 1, device=device)
            return envelope_pred, weights
            
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
            eeg: Input EEG of shape (batch, seq_len_prediction, channels)
            envelope_context: Input envelope of shape (batch, seq_len_context, 1)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope (batch, seq_len_prediction, 1)
                'envelope_branch': Envelope branch predictions (batch, seq_len_prediction, 1)
                'eeg_branch': EEG branch predictions (batch, seq_len_prediction, 1)
                'fusion_weights': Fusion weights when applicable (shape depends on fusion strategy)
        """
        batch_size, seq_len_prediction, _ = eeg.shape
        device = eeg.device

        envelope_context = envelope_context[:,32:,:]

        print("Forward pass of the model")
        print(f"EEG shape: {eeg.shape}")
        print(f"Envelope context shape: {envelope_context.shape}")
        
        # Use predict_autoregressive to generate the full sequence
        full_envelope_sequence = self.envelope_branch(envelope_context=envelope_context)
        
        # The envelope prediction is the full sequence
        envelope_pred = full_envelope_sequence['envelope_pred']

        # Apply sigmoid to the envelope prediction to ensure it is between 0 and 1
        #envelope_pred = torch.sigmoid(envelope_pred)

        # scale the envelope prediction to be between 0 and 1
        # inside forward()
        eps = 1e-6
        # compute per‑batch (or per‑sample) min and max
        mn = envelope_pred.min(dim=1, keepdim=True).values
        mx = envelope_pred.max(dim=1, keepdim=True).values

        # scale into [0,1]
        envelope_pred = (envelope_pred - mn) / (mx - mn + eps)

        # For envelope-only strategy, skip EEG branch computation
        if self.fusion_strategy == 'envelope_only':
            # Return envelope prediction as the final output
            # Set eeg_pred to None or zeros for consistency in return structure
            eeg_pred = torch.zeros_like(envelope_pred)
            
            # Use fuse_predictions to get both the fused predictions and weights
            fused_predictions, fusion_weights = self.fuse_predictions(
                envelope_pred=envelope_pred,
                eeg_pred=eeg_pred
            )
            
            return {
                'envelope': fused_predictions,
                'envelope_branch': envelope_pred,
                'eeg_branch': eeg_pred,
                'fusion_weights': fusion_weights
            }
        
        # For early fusion, we concatenate the envelope prediction with the EEG data
        if self.fusion_strategy == 'early_f':
            # Concatenate along the channel dimension
            # eeg: (batch, seq_len, channels)
            # envelope_pred: (batch, seq_len, 1)
            combined_input = torch.cat([eeg, envelope_pred], dim=2)
            
            # Process through EEG branch with the combined input
            eeg_branch_outputs = self.eeg_branch(combined_input)
            eeg_pred = eeg_branch_outputs['envelope']
        else:
            # Process through EEG branch normally
            eeg_branch_outputs = self.eeg_branch(eeg)
            eeg_pred = eeg_branch_outputs['envelope']

            # scale the eeg prediction to be between 0 and 1
            eps = 1e-6
            mn = eeg_pred.min(dim=1, keepdim=True).values
            mx = eeg_pred.max(dim=1, keepdim=True).values
            eeg_pred = (eeg_pred - mn) / (mx - mn + eps)
            
        # Fuse predictions and get weights
        fused_predictions, fusion_weights = self.fuse_predictions(
            envelope_pred=envelope_pred,
            eeg_pred=eeg_pred
        )
        
        return {
            'envelope': fused_predictions,
            'envelope_branch': envelope_pred,
            'eeg_branch': eeg_pred,
            'fusion_weights': fusion_weights
        }
    
    def forward_recursive(
        self,
        eeg: torch.Tensor,
        envelope_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            eeg: Input EEG of shape (batch, seq_len_prediction, channels)
            envelope_context: Input envelope of shape (batch, seq_len_context, 1)
            
        Returns:
            Dictionary containing:
                'envelope': Full predicted envelope (batch, seq_len_prediction, 1)
                'envelope_branch': Envelope branch predictions (batch, seq_len_prediction, 1)
                'eeg_branch': EEG branch predictions (batch, seq_len_prediction, 1)
                'fusion_weights': Fusion weights when applicable (shape depends on fusion strategy)
        """
        batch_size, seq_len_prediction, _ = eeg.shape
        device = eeg.device

        # predict the envelope context from the eeg context
        #envelope_context = self.eeg_branch(eeg_context)['envelope']
        #print(f"Envelope context shape: {envelope_context.shape}")
        #print(f"EEG context shape: {eeg_context.shape}")
        
        # get the envelope context length
        context_len = int(self.envelope_branch.context_seconds*self.envelope_branch.sr)
        # get the last context_len samples from the envelope_context
        print("Context length: ", context_len)
        #envelope_context = envelope_context[:, -context_len:, :]

        predicted_envelope_context = self.envelope_branch(envelope_context=envelope_context[:,-2*context_len:-context_len,:])['envelope_pred']
        envelope_context = predicted_envelope_context[:,:context_len,:]
        
        # Use predict_autoregressive to generate the full sequence
        full_envelope_sequence = self.envelope_branch(envelope_context=envelope_context)
        
        # The envelope prediction is the full sequence
        envelope_pred = full_envelope_sequence['envelope_pred']

        # Apply sigmoid to the envelope prediction to ensure it is between 0 and 1
        #envelope_pred = torch.sigmoid(envelope_pred)

        # scale the envelope prediction to be between 0 and 1
        # inside forward()
        eps = 1e-6
        # compute per‑batch (or per‑sample) min and max
        mn = envelope_pred.min(dim=1, keepdim=True).values
        mx = envelope_pred.max(dim=1, keepdim=True).values

        # scale into [0,1]
        envelope_pred = (envelope_pred - mn) / (mx - mn + eps)

        # For envelope-only strategy, skip EEG branch computation
        if self.fusion_strategy == 'envelope_only':
            # Return envelope prediction as the final output
            # Set eeg_pred to None or zeros for consistency in return structure
            eeg_pred = torch.zeros_like(envelope_pred)
            
            # Use fuse_predictions to get both the fused predictions and weights
            fused_predictions, fusion_weights = self.fuse_predictions(
                envelope_pred=envelope_pred,
                eeg_pred=eeg_pred
            )
            
            return {
                'envelope': fused_predictions,
                'envelope_branch': envelope_pred,
                'eeg_branch': eeg_pred,
                'fusion_weights': fusion_weights
            }
        
        # For early fusion, we concatenate the envelope prediction with the EEG data
        if self.fusion_strategy == 'early_f':
            # Concatenate along the channel dimension
            # eeg: (batch, seq_len, channels)
            # envelope_pred: (batch, seq_len, 1)
            combined_input = torch.cat([eeg, envelope_pred], dim=2)
            
            # Process through EEG branch with the combined input
            eeg_branch_outputs = self.eeg_branch(combined_input)
            eeg_pred = eeg_branch_outputs['envelope']
        else:
            # Process through EEG branch normally
            eeg_branch_outputs = self.eeg_branch(eeg)
            eeg_pred = eeg_branch_outputs['envelope']

            # scale the eeg prediction to be between 0 and 1
            eps = 1e-6
            mn = eeg_pred.min(dim=1, keepdim=True).values
            mx = eeg_pred.max(dim=1, keepdim=True).values
            eeg_pred = (eeg_pred - mn) / (mx - mn + eps)
            
        # Fuse predictions and get weights
        fused_predictions, fusion_weights = self.fuse_predictions(
            envelope_pred=envelope_pred,
            eeg_pred=eeg_pred
        )
        
        return {
            'envelope': fused_predictions,
            'envelope_branch': envelope_pred,
            'eeg_branch': eeg_pred,
            'fusion_weights': fusion_weights
        }