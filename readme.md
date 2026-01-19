# A Space-Time Transformer for Precipitation Forecasting
---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2511.11090) [![NeurIPS](https://img.shields.io/badge/NeurIPS_2025-🏆_1st_Place_CUMSUM-4b44ce.svg)](https://neurips.cc/virtual/2025/loc/san-diego/135896)

Levi Harris, Tianlong Chen

*The University of North Carolina at Chapel Hill*

### Weights

> [Google Drive](https://drive.google.com/drive/folders/1KOeIE1M5zVCTtmFwbAQH9yUo9vI_bbFk?usp=sharing)

*Download the weights above, then drag and drop the `weights` folder into this repo to use the code below as is.*

```
* src
* weights
    * sf-64-cls.pt
```

### Demo

```python
pip install torch einops
```

```python
import torch
import warnings

from src.model.SaTformer.SaTformer import SaTformer
from src.dataloader.challenge_one_dataloader import Sat2RadDataset

# quiet some annoying UserWarnings thrown by xarray 
# when opening datasets with phony_dims=None
warnings.simplefilter("ignore")

model = SaTformer(
    dim=512,
    num_frames=4,       # number HRIT input frames
    num_classes=64,     # number precipitation bins to use
    image_size=32,      # HRIT input spatial dimensions
    patch_size=4,
    channels=11,        # number HRIT radiance channels
    depth=12,           # number transformer encoder blocks
    heads=8,
    dim_head=64,
    attn_dropout=0.1,
    ff_dropout=0.1,
    rotary_emb=False,   # i.e., use postitional embeds
    attn="ST^2"
)

# NOTE: change to the path of YOUR model weights
WEIGHTS_FP = "weights/sf-64-cls.pt"

model.load_state_dict(torch.load(WEIGHTS_FP, weights_only=True), strict=False);
model.eval()

with torch.no_grad():
    inputs = torch.rand(1, 4, 11, 32, 32) # randomly generated HRIT input
    logits = model(inputs)                # call model forward pass
    print(logits.shape)                   # -> [1, 64]; raw model probs over output classes
```

![/assets/over-under-pred.png](/assets/over-under-pred.png)

### Citation

If you use this code in your research, please cite:

```bibtex
@article{harris2025satformer,
  title={A Space-Time Transformer for Precipitation Forecasting},
  author={Harris, Levi and Chen, Tianlong},
  journal={arXiv preprint arXiv:2511.11090},
  year={2025}
}
```

*Model predicted cumulative mass function (CMF) for a random input.*
