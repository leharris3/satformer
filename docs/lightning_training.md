# SaTformer PyTorch Lightning Training Guide

This guide covers training the SaTformer model using PyTorch Lightning with Weights & Biases (WandB) integration.

## Prerequisites

Install the additional dependency:

```bash
pip install pytorch-lightning
```

Or add to your environment:

```bash
uv add pytorch-lightning
```

## Quick Start

### Basic Training

```bash
python train_lightning.py
```

### Training with Custom Parameters

```bash
python train_lightning.py \
    --batch_size 64 \
    --lr 1e-4 \
    --max_epochs 100 \
    --wandb_project my_satformer_project
```

### Training without WandB

```bash
python train_lightning.py --no_wandb
```

## Command Line Arguments

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epochs` | 200 | Maximum training epochs |
| `--batch_size` | 128 | Training batch size |
| `--lr` | 1e-5 | Learning rate |
| `--weight_decay` | 0.0 | AdamW weight decay |
| `--num_workers` | 8 | DataLoader workers |
| `--precision` | 16-mixed | Training precision (32, 16-mixed, bf16-mixed) |
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps |

### Model Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dim` | 512 | Embedding dimension |
| `--depth` | 12 | Number of transformer layers |
| `--heads` | 8 | Number of attention heads |
| `--num_classes` | 64 | Number of precipitation bins |
| `--attn` | ST^2 | Attention type (T->S, S->T, ST, ST^2) |
| `--attn_dropout` | 0.1 | Attention dropout |
| `--ff_dropout` | 0.1 | Feed-forward dropout |

### Logging Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb_project` | satformer | WandB project name |
| `--wandb_entity` | None | WandB entity (team/username) |
| `--run_name` | None | Custom run name |
| `--no_wandb` | False | Disable WandB logging |

### Checkpoint Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | checkpoints | Directory for saving checkpoints |
| `--resume_from` | None | Path to checkpoint to resume from |

## Examples

### Multi-GPU Training

PyTorch Lightning automatically detects available GPUs:

```bash
python train_lightning.py --batch_size 256
```

For specific GPU selection:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py
```

### Resume Training

```bash
python train_lightning.py --resume_from checkpoints/last.ckpt
```

### Memory-Efficient Training

For limited GPU memory, reduce batch size and use gradient accumulation:

```bash
python train_lightning.py \
    --batch_size 32 \
    --accumulate_grad_batches 4 \
    --precision 16-mixed
```

### Hyperparameter Sweep

```bash
# Example: Testing different attention mechanisms
for attn in "T->S" "S->T" "ST" "ST^2"; do
    python train_lightning.py \
        --attn "$attn" \
        --run_name "satformer-$attn" \
        --max_epochs 50
done
```

## Metrics

The training script logs the following metrics:

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss on training data |
| `train/csi` | Critical Success Index (weather metric) |
| `train/f1` | Multi-class F1 score |
| `train/crps` | Continuous Ranked Probability Score |
| `val/loss` | Cross-entropy loss on validation data |
| `val/csi` | Validation CSI |
| `val/f1` | Validation F1 |
| `val/crps` | Validation CRPS |

## Model Checkpoints

Checkpoints are saved to `checkpoints/` (configurable) with the naming format:

```
satformer-{epoch}-{val_loss}.ckpt
```

The script saves:
- Top 3 checkpoints by validation loss
- `last.ckpt` for resuming

## Loading a Trained Model

```python
from train_lightning import SaTformerLightning

# Load from checkpoint
model = SaTformerLightning.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# Inference
import torch
x = torch.randn(1, 4, 11, 32, 32)  # [B, T, C, H, W]
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
```

## WandB Integration

When WandB is enabled, the following are logged:
- All hyperparameters
- Training and validation metrics
- Learning rate schedule
- Model gradients (every 100 steps)

View your runs at: https://wandb.ai/

### WandB Setup

1. Create an account at https://wandb.ai/
2. Run `wandb login` and enter your API key
3. Run training with `--wandb_project your_project`

## Architecture Details

The SaTformer model uses:
- **Input**: `[B, 4, 11, 32, 32]` - 4 frames of 11-channel satellite imagery
- **Output**: `[B, num_classes]` - logits over precipitation bins
- **Attention**: Configurable space-time attention (default: ST^2)
- **Optimizer**: AdamW with cosine annealing LR schedule
