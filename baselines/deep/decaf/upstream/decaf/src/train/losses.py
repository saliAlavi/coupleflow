import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import namedtuple

class PearsonLoss(nn.Module):
    """
    Pearson correlation coefficient loss.
    Loss = 1 - correlation (to convert correlation to a minimization problem)
    """
    def __init__(self, reduction: str = 'mean', as_metric: bool = False):
        """
        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.reduction = reduction
        self.as_metric = as_metric

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Pearson correlation loss.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: 1 - correlation (to make it a minimization problem)
        """
        # Flatten the tensors to (batch_size, seq_len)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        if mask is not None:
            # Flatten mask to match x and y
            mask = mask.view(mask.size(0), -1)
            
            # Apply mask
            x_masked = x * mask.float()
            y_masked = y * mask.float()
            
            # Compute mean only over masked positions
            mask_sum = mask.sum(dim=1, keepdim=True).float()
            x_mean = (x_masked.sum(dim=1, keepdim=True) / mask_sum)
            y_mean = (y_masked.sum(dim=1, keepdim=True) / mask_sum)
            
            # Center the variables
            x_centered = (x - x_mean) * mask.float()
            y_centered = (y - y_mean) * mask.float()
            
            # Compute correlation for each sample in batch (only for masked positions)
            x_std = torch.sqrt(torch.sum(x_centered**2, dim=1, keepdim=True) + 1e-8)
            y_std = torch.sqrt(torch.sum(y_centered**2, dim=1, keepdim=True) + 1e-8)
            corr = torch.sum(x_centered * y_centered, dim=1, keepdim=True) / (x_std * y_std)
        else:
            # Original implementation for unmasked case
            x = x - x.mean(dim=1, keepdim=True)
            y = y - y.mean(dim=1, keepdim=True)
            
            x_std = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-8)
            y_std = torch.sqrt(torch.sum(y**2, dim=1, keepdim=True) + 1e-8)
            corr = torch.sum(x * y, dim=1, keepdim=True) / (x_std * y_std)
        
        # Convert to loss (1 - correlation)
        if self.as_metric:
            loss = corr
        else:
            loss = 1 - corr
            
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class PearsonL1Loss(nn.Module):
    """
    Combined loss function that adds weighted L1 loss to Pearson correlation loss.
    Loss = (1 - correlation) + alpha * L1_loss
    """
    def __init__(self, alpha: float = 0.2, reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for L1 loss component
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction='none')
        self.pearson_loss = PearsonLoss(reduction='none')
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute combined Pearson correlation and L1 loss.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: (1 - correlation) + alpha * L1_loss
        """
        pearson_loss = self.pearson_loss(x, y, mask)  # Shape: (batch_size, 1)
        
        # Compute L1 loss only on masked positions if mask is provided
        x_flat = x.view(x.size(0), -1)  # Shape: (batch_size, seq_len)
        y_flat = y.view(y.size(0), -1)
        
        if mask is not None:
            mask_flat = mask.view(mask.size(0), -1).float()
            l1_losses = self.l1_loss(x_flat, y_flat) * mask_flat
            # Average L1 loss over masked positions only
            l1_loss = (l1_losses.sum(dim=1) / (mask_flat.sum(dim=1) + 1e-8)).unsqueeze(1)
        else:
            l1_loss = self.l1_loss(x_flat, y_flat).mean(dim=1, keepdim=True)
        
        # Combine the losses
        combined_loss = pearson_loss + self.alpha * l1_loss
        
        # Apply reduction over batch dimension
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss

class MaskedMSELoss(nn.Module):
    """
    MSE loss that can be computed only on masked positions.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute MSE loss, optionally only on masked positions.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: MSE loss
        """
        # mask=None # TODO: remove this
        # Flatten the tensors
        x_flat = x.view(x.size(0), -1)  # Shape: (batch_size, seq_len)
        y_flat = y.view(y.size(0), -1)
        
        # Compute element-wise MSE
        element_wise_mse = self.mse_loss(x_flat, y_flat)
        
        if mask is not None:
            # Flatten mask and convert to float
            mask_flat = mask.view(mask.size(0), -1).float()
            
            # Apply mask to losses
            masked_losses = element_wise_mse * mask_flat
            
            # Average losses over masked positions only for each sample
            sample_losses = (masked_losses.sum(dim=1) / (mask_flat.sum(dim=1) + 1e-8))
        else:
            # If no mask, average over all positions
            sample_losses = element_wise_mse.mean(dim=1)
        
        # Apply reduction over batch dimension
        if self.reduction == 'mean':
            return sample_losses.mean()
        elif self.reduction == 'sum':
            return sample_losses.sum()
        else:  # 'none'
            return sample_losses

class MaskedL1Loss(nn.Module):
    """
    L1 loss that can be computed only on masked positions.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute L1 loss, optionally only on masked positions.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: L1 loss
        """
        # Flatten the tensors
        x_flat = x.view(x.size(0), -1)  # Shape: (batch_size, seq_len)
        y_flat = y.view(y.size(0), -1)
        
        # Compute element-wise L1
        element_wise_l1 = self.l1_loss(x_flat, y_flat)
        
        if mask is not None:
            # Flatten mask and convert to float
            mask_flat = mask.view(mask.size(0), -1).float()
            
            # Apply mask to losses
            masked_losses = element_wise_l1 * mask_flat
            
            # Average losses over masked positions only for each sample
            sample_losses = (masked_losses.sum(dim=1) / (mask_flat.sum(dim=1) + 1e-8))
        else:
            # If no mask, average over all positions
            sample_losses = element_wise_l1.mean(dim=1)
        
        # Apply reduction over batch dimension
        if self.reduction == 'mean':
            return sample_losses.mean()
        elif self.reduction == 'sum':
            return sample_losses.sum()
        else:  # 'none'
            return sample_losses

class DistanceCorrelationLoss(nn.Module):
    """
    Distance Correlation loss function.
    Loss = 1 - distance_correlation (to convert correlation to a minimization problem)
    
    Distance Correlation measures how independent two samples are, with values
    in [0, 1], and with zero when the two samples are independent.
    Distance variance measures how close to identical a set of observations are.
    """
    def __init__(self, reduction: str = 'mean', as_metric: bool = False):
        """
        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
            as_metric: If True, returns the correlation instead of the loss
        """
        super().__init__()
        self.reduction = reduction
        self.as_metric = as_metric
    
    def _distance_correlation(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculate the distance correlation between x and y.
        
        Args:
            x: Tensor of shape (..., N, F_x). N observations, F_x measurements.
            y: Tensor of shape (..., N, F_y). N observations, F_y measurements.
        
        Returns:
            namedtuple with elements (dcor, dcov, dvar_x, dvar_y)
        """
        N, _ = x.shape[-2:]
        center = lambda arr: (
            arr
            - arr.mean(-2, keepdim=True)
            - arr.mean(-1, keepdim=True)
            + arr.mean((-1, -2), keepdim=True)
        )
        # get centered pairwise distances
        x_centered = center(torch.cdist(x, x, 2))
        dvar_x = torch.sqrt((x_centered * x_centered).mean((-1, -2), keepdim=True))
        
        # For y
        y_centered = center(torch.cdist(y, y, 2))
        # compute relevant statistics
        dcov = torch.sqrt((x_centered * y_centered).mean((-1, -2), keepdim=True))
        dvar_y = torch.sqrt((y_centered * y_centered).mean((-1, -2), keepdim=True))
        dcor = dcov / torch.sqrt(dvar_x * dvar_y + 1e-8)
        
        # Avoid inplace operations with squeeze_() which cause gradient issues
        dcov = dcov.squeeze(-1).squeeze(-1)
        dvar_y = dvar_y.squeeze(-1).squeeze(-1)
        dcor = dcor.squeeze(-1).squeeze(-1)
        dvar_x = dvar_x.squeeze(-1).squeeze(-1)
        
        return namedtuple("DistanceCorrelation", ["dcor", "dcov", "dvar_x", "dvar_y"])(
            dcor, dcov, dvar_x, dvar_y
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute distance correlation loss.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: 1 - distance_correlation (to make it a minimization problem)
                          or distance_correlation if as_metric=True
        """
        # Reshape tensors for distance correlation computation
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, 1)  # (batch_size, seq_len, 1)
        y_reshaped = y.view(batch_size, -1, 1)  # (batch_size, seq_len, 1)
        
        if mask is not None:
            # Apply mask if provided
            mask_expanded = mask.view(batch_size, -1, 1).float()
            x_masked = x_reshaped * mask_expanded
            y_masked = y_reshaped * mask_expanded
            
            # Process each sample in the batch individually
            batch_corrs = []
            for i in range(batch_size):
                # Get only the masked positions for this sample
                sample_mask = mask[i].bool()
                if sample_mask.sum() > 1:  # Need at least 2 points for correlation
                    sample_x = x_reshaped[i, sample_mask]
                    sample_y = y_reshaped[i, sample_mask]
                    # Ensure proper shape for distance calculation
                    sample_x = sample_x.unsqueeze(0)  # Add batch dimension
                    sample_y = sample_y.unsqueeze(0)
                    corr = self._distance_correlation(sample_x, sample_y).dcor
                    batch_corrs.append(corr)
                else:
                    # If no valid points, add zero correlation
                    batch_corrs.append(torch.tensor(0.0, device=x.device))
            
            # Stack results
            correlations = torch.stack(batch_corrs)
        else:
            # Calculate for the whole batch
            results = []
            for i in range(batch_size):
                sample_x = x_reshaped[i:i+1]  # Keep batch dimension
                sample_y = y_reshaped[i:i+1]
                results.append(self._distance_correlation(sample_x, sample_y).dcor)
            correlations = torch.stack(results)
        
        # Convert to loss if needed
        if self.as_metric:
            loss = correlations
        else:
            loss = 1 - correlations
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
class DistanceCorrelationL1Loss(nn.Module):
    """
    Combined loss function that adds weighted L1 loss to Distance Correlation loss.
    Loss = (1 - distance_correlation) + alpha * L1_loss
    """
    def __init__(self, alpha: float = 0.2, reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for L1 loss component
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction='none')
        self.dcor_loss = DistanceCorrelationLoss(reduction='none')
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute combined Distance Correlation and L1 loss.
        
        Args:
            x: Predicted values (batch_size, seq_len, 1)
            y: Target values (batch_size, seq_len, 1)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates positions to compute loss
            
        Returns:
            torch.Tensor: (1 - distance_correlation) + alpha * L1_loss
        """
        dcor_loss = self.dcor_loss(x, y, mask)  # Shape: (batch_size,) or scalar
        
        # Compute L1 loss only on masked positions if mask is provided
        x_flat = x.view(x.size(0), -1)  # Shape: (batch_size, seq_len)
        y_flat = y.view(y.size(0), -1)
        
        if mask is not None:
            mask_flat = mask.view(mask.size(0), -1).float()
            l1_losses = self.l1_loss(x_flat, y_flat) * mask_flat
            # Average L1 loss over masked positions only
            l1_loss = (l1_losses.sum(dim=1) / (mask_flat.sum(dim=1) + 1e-8))
        else:
            l1_loss = self.l1_loss(x_flat, y_flat).mean(dim=1)
        
        # Combine the losses
        combined_loss = dcor_loss + self.alpha * l1_loss
        
        # Apply reduction over batch dimension
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss
        

