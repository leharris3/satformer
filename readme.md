# A Space-Time Transformer for Precipitation Nowcasting
---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2511.11090) [![NeurIPS](https://img.shields.io/badge/NeurIPS_2025-🏆_1st_Place_CUMSUM-4b44ce.svg)](https://neurips.cc/virtual/2025/loc/san-diego/135896)

Levi Harris, Tianlong Chen

*The University of North Carolina at Chapel Hill*

### Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# sync dependencies
uv sync
```

### Weights

Download pretrained weights from [HuggingFace](https://huggingface.co/leharris3/satformer):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="leharris3/satformer", filename="sf-64-cls.pt", local_dir="weights")
```

### Demo

```python
import torch
import warnings

from huggingface_hub import hf_hub_download
from src.model.SaTformer.SaTformer import SaTformer

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

WEIGHTS_FP = hf_hub_download(repo_id="leharris3/satformer", filename="sf-64-cls.pt")

model.load_state_dict(torch.load(WEIGHTS_FP, weights_only=True), strict=False);
model.eval()

with torch.no_grad():
    inputs = torch.rand(1, 4, 11, 32, 32) # randomly generated HRIT input
    logits = model(inputs)                # call model forward pass
    print(logits.shape)                   # -> [1, 64]; raw model probs over output classes
```

***

<p align="center">
  <img src="/assets/over-under-pred.png" width="600">
  <br>
  <em>Model predicted cumulative mass function (CMF) for a random input.</em>
</p>

### Repository Structure

```
satformer/
├── train.py                    # training entrypoint
├── test.py                     # inference entrypoint
├── demo.ipynb                  # interactive demo notebook
├── configs/                    # training & test configs
├── scripts/                    # launcher scripts
└── src/
    ├── model/
    │   └── SaTformer/
    │       ├── SaTformer.py    # model architecture
    │       └── rotary.py       # rotary positional embeddings
    ├── dataloader/             # dataset & preprocessing
    └── util/                   # losses, metrics, logging, plotting
```

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
