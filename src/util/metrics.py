import torch
import numpy as np

from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure as ssim
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as psnr
from typing import Optional, Tuple


def RMSE_surface_roughness_l1(
    pred: torch.Tensor, target: torch.Tensor, dataset_min: float, dataset_max: float
) -> torch.Tensor:

    # unnormalize to original topology distribution
    pred   = pred   * (dataset_max - dataset_min) + dataset_min
    target = target * (dataset_max - dataset_min) + dataset_min

    def calculate_roughness(X: torch.Tensor) -> torch.Tensor:
        """
        Per Jayed's specs...
        """
        # Zero-centering the data
        X -= torch.mean(X)
        # RMS roughness in nm
        Sq = torch.sqrt(torch.mean(X**2)) * 1e9
        # Mean roughness in nm
        Sa = torch.mean(torch.abs(X)) * 1e9
        return Sq

    return (calculate_roughness(pred) - calculate_roughness(target)).abs()


def MAE(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (preds - target).abs().mean()


def MSE(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (preds - target).pow(2).mean()


def SSIM(
    preds: torch.Tensor,
    target: torch.Tensor,
    _data_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    ...
    """

    low, high = _data_range
    diff = high - low

    # check that all vals are in range
    if not ((preds >= low) & (preds <= high)).all():
        raise ValueError(
            f"Values in `preds` are out of the expected range [{low}, {high}]. "
            f"Detected min={preds.min().item()}, max={preds.max().item()}"
        )
    if not ((target >= low) & (target <= high)).all():
        raise ValueError(
            f"Values in `target` are out of the expected range [{low}, {high}]. "
            f"Detected min={target.min().item()}, max={target.max().item()}"
        )
    
    # val = ssim(preds.clamp(low, high).float(), target.clamp(low, high).float(), data_range=diff)
    
    # NOTE: don't use explicit range
    # this implementation calculates max, min values automatically
    val = ssim(preds, target)

    return val


def PSNR(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    ...
    """
    # check that all vals are in range
    low, high = data_range
    if not ((preds >= low) & (preds <= high)).all():
        raise ValueError(
            f"Values in `preds` are out of the expected range [{low}, {high}]. "
            f"Detected min={preds.min().item()}, max={preds.max().item()}"
        )
    if not ((target >= low) & (target <= high)).all():
        raise ValueError(
            f"Values in `target` are out of the expected range [{low}, {high}]. "
            f"Detected min={target.min().item()}, max={target.max().item()}"
        )
    
    # val = psnr(preds, target, data_range=high - low)

    # NOTE: don't use explicit range
    # this implementation calculates max, min values automatically
    val = psnr(preds, target)

    return val


def OLDER(y_char: dict, y_sparse_char: dict) -> float:
    """
    Offline Domain-Expert Rating.
    A weighted average of percent difference of characterization of two current-maps using Celano labs scripts.

    Parameters
    ---
    :y_char: dictionary output of celano lab script for y
    :y_sparse_char: dictionary output of celano lab script for y_sparse

    Returns
    ---
    :val float: mean-abs %-diff in range [0, inf)
    """

    # KEYS = ["average_surface_current", "coverage_percentage", "num_extended_shapes", "total_Area Extended Shapes"]
    diffs = []

    for k1, k2 in zip(y_char.keys(), y_sparse_char.keys()):
        # if k1 not in KEYS or k2 not in KEYS: continue
        val1, val2 = y_char[k1], y_sparse_char[k2]
        if val1 == 0 and val2 == 0:
            diffs.append(0.0)
        else:
            avg_val = (val1 + val2) / 2.0
            # || %diff || = || 2 * || val1 - val2 || / (val1 + val2) ||
            pdiff = abs(val1 - val2) / avg_val
            diffs.append(abs(pdiff))

    return np.mean(diffs)


if __name__ == "__main__":
    pass
