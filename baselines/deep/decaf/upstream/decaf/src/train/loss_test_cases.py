import torch
from losses import PearsonLoss, PearsonL1Loss, MaskedMSELoss, MaskedL1Loss
import numpy as np
# ----------------------------
# 1. PearsonLoss Test
# ----------------------------
from typing import Optional

def test_pearson_loss(pearson_loss_cls, 
                      x_vals: torch.Tensor, 
                      y_vals: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None, 
                      expected_loss: Optional[float] = None):
    """
    Generic test function for PearsonLoss.
    
    Args:
        pearson_loss_cls: The PearsonLoss class (or instance).
        x_vals: Input predictions (B, T, 1).
        y_vals: Ground truth (B, T, 1).
        mask: Optional mask (B, T).
        expected_loss: If provided, we compare the computed loss with this expected value.
    """
    loss_fn = pearson_loss_cls()
    loss_val = loss_fn(x_vals, y_vals, mask)
    print("PearsonLoss - computed value:", loss_val.item())
    if expected_loss is not None:
        print("Expected value:", expected_loss)
        assert torch.isclose(loss_val, torch.tensor(expected_loss, dtype=loss_val.dtype), atol=1e-6), \
            f"PearsonLoss mismatch: got {loss_val.item()}, expected {expected_loss}"


def test_pearson_loss_as_metric():
    """
    Demonstrates using PearsonLoss as a metric (i.e., returning correlation).
    """
    print("=== PearsonLoss as a Metric Test ===")
    
    # Import or define your PearsonLoss with `as_metric=True`
    from losses import PearsonLoss  # Update to your actual import path

    # 1) Perfect positive correlation
    #    x == y => correlation = +1
    x = torch.tensor([[[1.0], [2.0], [3.0]],
                      [[0.0], [1.0], [2.0]]])  # shape (2,3,1)
    y = x.clone()

    metric_fn = PearsonLoss(as_metric=True, reduction='mean')
    metric_val_1 = metric_fn(x, y)
    print("Case 1 (perfect positive corr) - computed metric:", metric_val_1.item())
    # Expect close to +1
    assert torch.isclose(metric_val_1, torch.tensor(1.0), atol=1e-6), \
        f"Expected correlation of ~1.0, but got {metric_val_1.item()}"

    # 2) Perfect negative correlation
    #    x = -y => correlation = -1
    x_neg = -x
    metric_val_2 = metric_fn(x_neg, y)
    print("Case 2 (perfect negative corr) - computed metric:", metric_val_2.item())
    # Expect close to -1
    assert torch.isclose(metric_val_2, torch.tensor(-1.0), atol=1e-6), \
        f"Expected correlation of ~-1.0, but got {metric_val_2.item()}"

    print("All PearsonLoss metric tests passed!")



def run_pearson_loss_tests():
    from losses import PearsonLoss  # Adjust import to match your file structure

    print("=== PearsonLoss Tests ===")
    
    # Test 1: Perfect correlation => correlation=1 => loss = 1 - 1 = 0
    # x = y exactly
    x = torch.tensor([[[1.0],[2.0],[3.0]],
                      [[2.0],[4.0],[6.0]]])  # shape (2, 3, 1)
    y = x.clone()
    test_pearson_loss(PearsonLoss, x, y, expected_loss=0.0)
    
    # Test 2: Perfect correlation but partially masked
    # Mask out some positions (the correlation among the unmasked positions is still perfect)
    mask = torch.tensor([[True,  True,  False],
                         [False, True,  True]])
    # Since x=y, among the masked entries, correlation is 1 => loss=0
    test_pearson_loss(PearsonLoss, x, y, mask, expected_loss=0.0)
    
    # Test 3: Invert sign => correlation ~ -1 => loss=1 - (-1)=2
    x_neg = -x
    test_pearson_loss(PearsonLoss, x_neg, y, expected_loss=2.0)
    
    print()

# ----------------------------
# 2. PearsonL1Loss Test
# ----------------------------
def test_pearson_l1_loss(pearson_l1_loss_cls, 
                         x_vals: torch.Tensor, 
                         y_vals: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None, 
                         alpha: float = 0.2,
                         expected_loss: Optional[float] = None):
    """
    Generic test function for PearsonL1Loss.
    """
    loss_fn = pearson_l1_loss_cls(alpha=alpha)
    loss_val = loss_fn(x_vals, y_vals, mask)
    print("PearsonL1Loss - computed value:", loss_val.item())
    if expected_loss is not None:
        print("Expected value:", expected_loss)
        assert torch.isclose(loss_val, torch.tensor(expected_loss, dtype=loss_val.dtype), atol=1e-6), \
            f"PearsonL1Loss mismatch: got {loss_val.item()}, expected {expected_loss}"


def run_pearson_l1_loss_tests():
    from losses import PearsonL1Loss  # Adjust import to match your file structure

    print("=== PearsonL1Loss Tests ===")

    # Example data
    x = torch.tensor([[[1.0],[2.0],[3.0]],
                      [[2.0],[4.0],[6.0]]])  # shape (2, 3, 1)
    y = torch.tensor([[[1.0],[2.0],[3.0]],
                      [[1.0],[3.0],[5.0]]])  # shape (2, 3, 1)
    # For the second sample, there's a constant offset of +1 for each element.
    # Pearson correlation for each sample:
    #   1st sample is perfectly correlated => 1 - 1 = 0
    #   2nd sample x=[2,4,6], y=[1,3,5]
    #       correlation should be 1.0 as well (x is y+1, which is still a perfect linear transform).
    # So Pearson part => 0 for both samples => average => 0
    
    # L1 part:
    #   1st sample: L1 = 0
    #   2nd sample: difference of 1 for each of the 3 positions => L1=1 => mean of 1 => total for second sample
    # Weighted by alpha=0.2 => 0.2 * 1 = 0.2
    # Then average across 2 samples => 0.1
    # NOTE: Because we do "per-sample" first, then average:
    #   1st sample total = 0 (corr) + 0 (L1) = 0
    #   2nd sample total = 0 (corr) + 0.2 (L1) = 0.2
    #   average = (0 + 0.2)/2 = 0.1

    test_pearson_l1_loss(PearsonL1Loss, x, y, alpha=0.2, expected_loss=0.1)
    
    # Test with a mask
    mask = torch.tensor([[True, True, False],
                         [True, False, True]])
    # We'll just print the loss. You can manually compute if desired.
    test_pearson_l1_loss(PearsonL1Loss, x, y, mask, alpha=0.2)

    print()

# ----------------------------
# 3. MaskedMSELoss Test
# ----------------------------
def test_masked_mse_loss(masked_mse_cls, 
                         x_vals: torch.Tensor, 
                         y_vals: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None, 
                         expected_loss: Optional[float] = None):
    """
    Generic test function for MaskedMSELoss.
    """
    loss_fn = masked_mse_cls()
    loss_val = loss_fn(x_vals, y_vals, mask)
    print("MaskedMSELoss - computed value:", loss_val.item())
    if expected_loss is not None:
        print("Expected value:", expected_loss)
        assert torch.isclose(loss_val, torch.tensor(expected_loss, dtype=loss_val.dtype), atol=1e-6), \
            f"MaskedMSELoss mismatch: got {loss_val.item()}, expected {expected_loss}"


def run_masked_mse_loss_tests():
    from losses import MaskedMSELoss  # Adjust import to match your file structure

    print("=== MaskedMSELoss Tests ===")

    # Test 1: x = y => MSE=0
    x = torch.tensor([[[1.0],[2.0],[3.0]]])
    y = torch.tensor([[[1.0],[2.0],[3.0]]])
    test_masked_mse_loss(MaskedMSELoss, x, y, None, expected_loss=0.0)
    
    # Test 2: partial differences + mask
    # shape: (1,3,1)
    x = torch.tensor([[[1.0],[4.0],[2.0]]])
    y = torch.tensor([[[2.0],[2.0],[2.0]]])
    # MSE per position:
    #   pos0: (1 - 2)^2 = 1
    #   pos1: (4 - 2)^2 = 4
    #   pos2: (2 - 2)^2 = 0
    #
    # Mask out pos1 => only compute MSE on pos0, pos2
    mask = torch.tensor([[True, False, True]])
    # MSE among those masked positions => avg(1, 0) = 0.5
    test_masked_mse_loss(MaskedMSELoss, x, y, mask, expected_loss=0.5)

    print()

# ----------------------------
# 4. MaskedL1Loss Test
# ----------------------------
def test_masked_l1_loss(masked_l1_cls, 
                        x_vals: torch.Tensor, 
                        y_vals: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None, 
                        expected_loss: Optional[float] = None):
    """
    Generic test function for MaskedL1Loss.
    """
    loss_fn = masked_l1_cls()
    loss_val = loss_fn(x_vals, y_vals, mask)
    print("MaskedL1Loss - computed value:", loss_val.item())
    if expected_loss is not None:
        print("Expected value:", expected_loss)
        assert torch.isclose(loss_val, torch.tensor(expected_loss, dtype=loss_val.dtype), atol=1e-6), \
            f"MaskedL1Loss mismatch: got {loss_val.item()}, expected {expected_loss}"


def run_masked_l1_loss_tests():
    from losses import MaskedL1Loss  # Adjust import to match your file structure

    print("=== MaskedL1Loss Tests ===")

    # Test 1: x = y => L1=0
    x = torch.tensor([[[1.0],[2.0],[3.0]]])
    y = torch.tensor([[[1.0],[2.0],[3.0]]])
    test_masked_l1_loss(MaskedL1Loss, x, y, None, expected_loss=0.0)
    
    # Test 2: partial differences + mask
    x = torch.tensor([[[1.0],[4.0],[2.0]]])
    y = torch.tensor([[[2.0],[2.0],[2.0]]])
    # L1 per position:
    #   pos0: |1 - 2| = 1
    #   pos1: |4 - 2| = 2
    #   pos2: |2 - 2| = 0
    #
    # Mask out pos1 => only compute L1 on pos0, pos2 => avg(1, 0) = 0.5
    mask = torch.tensor([[True, False, True]])
    test_masked_l1_loss(MaskedL1Loss, x, y, mask, expected_loss=0.5)

    print()

if __name__ == "__main__":
    """
    Example script demonstrating the tests.
    NOTE: In your actual project, replace "from losses import ..." with the appropriate
    import lines for each custom loss class or place all in one file.
    """
    run_pearson_loss_tests()
    test_pearson_loss_as_metric()
    run_pearson_l1_loss_tests()
    run_masked_mse_loss_tests()
    run_masked_l1_loss_tests()

    import numpy as np
    from scipy.stats import pearsonr

    def pearson_numpy(x_np, y_np, mask_np=None):
        if mask_np is not None:
            # Use only masked positions
            x_masked = x_np[mask_np]
            y_masked = y_np[mask_np]
            return pearsonr(x_masked, y_masked)[0]
        else:
            return pearsonr(x_np, y_np)[0]

    # Example usage:
    x_torch = torch.randn(1, 3, 1)  # Random tensor with shape (1,3,1)
    y_torch = torch.randn(1, 3, 1)  # Random tensor with shape (1,3,1) 
    mask_torch = torch.tensor([[[False], [True], [True]]])  # for instance

    # Flatten appropriately:
    x_np = x_torch.view(-1).numpy()
    y_np = y_torch.view(-1).numpy()
    mask_np = mask_torch.view(-1).numpy().astype(bool)

    corr_scipy = pearson_numpy(x_np, y_np, mask_np)
    corr_torch = PearsonLoss(as_metric=True)(x_torch, y_torch, mask_torch).item()

    print("Pearson corr (SciPy)  :", corr_scipy)
    print("Pearson corr (PyTorch):", corr_torch)

