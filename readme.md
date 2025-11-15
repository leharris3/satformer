# A Space-Time Transformer for Precipitation Forecasting
---

**Levi Harris**, Tianlong Chen

*The Unviersity of North Carolina at Chapel Hill*

```python
import torch
from src.model.SaTformer.SaTformer import SaTformer


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

WEIGHTS_FP = "weights/sf-64-cls.pt"

model.load_state_dict(torch.load(WEIGHTS_FP, weights_only=True), strict=False);
model.eval()

# predict using a randomly generated HRIT sequence
HRIT_dummy_input = torch.rand(1, 4, 11, 32, 32)
with torch.no_grad():
    y = model(HRIT_dummy_input)
    print(y.shape)
```