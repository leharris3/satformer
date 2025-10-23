from signal import siginterrupt
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from typing import List


# https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def apply_color_palette(map_like: np.ndarray) -> np.ndarray:
    """
    Applies a color palette to a 2D input array (H, W) and returns
    an RGB image (H, W, C).

    Parameters:
    -----------
    map_tensor : np.ndarray
        A  NumPy array representing
        some feature map or heatmap data.

    Returns:
    --------
    colored_image : np.ndarray
        An array of shape (H, W, C), where each pixel has RGB values
        in the [0, 1] range.
    """
    
    map_like = map_like.astype(np.float32)
    min_val, max_val = np.min(map_like), np.max(map_like)
    
    # [-1, 1] -> [0, 1]
    # NOTE: this re-normalization step is needed for models like ControlNet
    normalized_map = (map_like - min_val) / (max_val - min_val)
    
    # normalized_map = map_like
    
    cmap = cm.get_cmap("viridis")
    colored_map = cmap(normalized_map)
    
    # HACK:
    if len(colored_map.shape) == 4:
        colored_image = colored_map[:, :, :, 0]
    else:
        colored_image = colored_map[..., :3]
    return colored_image


def convert_to_img_like(*args: torch.Tensor) -> List[np.ndarray]:
    """
    Convert one or more tensors from any range to [0, 1].
    Cast to int, move to CPU, and return them all as NumPy arrays.
    """
    results = []
    for x in args:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        results.append(apply_color_palette(x))
    return results


def grayscale_to_2d(grayscale_like: torch.Tensor) -> torch.Tensor:
    """
    Converts a 'grayscale' tensor of shape (128, 128, 3)
    to shape (128, 128) by averaging across the last dimension.
    """
    tensor_2d = torch.mean(grayscale_like, axis=-1)
    return tensor_2d
