import torch
import numpy as np

from torch import Tensor
from scipy.stats import moment


def calc_moment_based_stats(samples: np.ndarray) -> dict:
    """
    Calculate a panel of moment-based stats for SPM samples with shape [B, N, M].
    - Ref: https://gwyddion.net/documentation/user-guide-en/statistical-analysis.html#stat-quantities 

    "Moment based quantities are expressed using integrals of the 
    height distribution function with some powers of height. 
    They include the familiar quantities"
    
    Returns
    ---
        1. Average value
        2. RMS roughnes (sq)
        3. RMS Mean roughness (Sa)
        4. Skew (Ssk)
        5. Excess kurtosis
    """

    assert len(samples.shape) == 3, f"Error: expected sample to have shape [B, N, M]"

    stats = {
        "mean"          : 0,
        "mean_rms"      : 0,
        "mean_roughness": 0,
        "skewness"      : 0,
        "kurtosis"      : 0,
    }

    for sample in samples:

        if isinstance(sample, Tensor):
            sample = sample.detach().cpu().numpy()

        sample = sample.flatten()

        # calculate central moments
        mu_2 = moment(sample, moment=2)
        mu_3 = moment(sample, moment=3)
        mu_4 = moment(sample, moment=4)

        average_value = sample.mean()

        # σ = μ_2 ^ 1/2
        rms = mu_2 ** .5
        mean_rms = rms.mean()
        mean_roughness = abs(sample - sample.mean()).mean()

        # γ_1 = (μ_3) / (μ_2 ^ (3/2))
        skewness = mu_3 / (mu_2 ** (3/2))

        # γ_2 = (μ_4) / (μ_2 ^ 2) - 3
        kurtosis = (mu_4 / (mu_2 ** 2)) - 3

        sample_stats = {
            "mean"          : average_value,
            "mean_rms"      : mean_rms,
            "mean_roughness": mean_roughness,
            "skewness"      : skewness,
            "kurtosis"      : kurtosis,
        }

        for k in stats:
            stats[k] += sample_stats[k]

    # calculate average stats
    for k in stats:
        stats[k] = (stats[k] / samples.shape[0])

    return stats


def cal_moment_based_errs(pred: np.ndarray, target: np.ndarray) -> dict:

    pred_stats = calc_moment_based_stats(pred)
    target_stats = calc_moment_based_stats(target)

    errs = {}
    for k in pred_stats:
        ps = pred_stats[k]
        ts = target_stats[k]
        err = abs(ps - ts)
        errs[k] = err

    return errs
    
        
if __name__ == "__main__":
    fp     = "/playpen/mufan/levi/tianlong-chen-lab/sparse-cafm/data/raw-data/3-12-25/A1 512_ Height_Backward_020.npy"
    
    s1 = Tensor(np.load(fp)).unsqueeze(0).numpy()
    s2 = s1 + s1.mean()
    
    from pprint import pprint
    pprint(calc_moment_based_stats(s1), indent=4)