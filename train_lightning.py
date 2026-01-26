"""
PyTorch Lightning training script for SaTformer.

Usage:
    python train_lightning.py
    python train_lightning.py --batch_size 64 --lr 1e-4
    python train_lightning.py --wandb_project my_project --max_epochs 100
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.model.SaTformer.SaTformer import SaTformer
from src.dataloader.challenge_one_dataloader import Sat2RadDataset
from src.util.eval_metrics import mean_csi, mean_f1, mean_crps


class SaTformerLightning(pl.LightningModule):
    """Lightning wrapper for SaTformer model."""

    def __init__(
        self,
        dim: int = 512,
        num_frames: int = 4,
        num_classes: int = 64,
        image_size: int = 32,
        patch_size: int = 4,
        channels: int = 11,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        rotary_emb: bool = False,
        attn: str = "ST^2",
        lr: float = 1e-5,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SaTformer(
            dim=dim,
            num_frames=num_frames,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            rotary_emb=rotary_emb,
            attn=attn,
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x = batch["X_norm"]
        y = batch["y_reg_norm_oh"]

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        csi = mean_csi(logits, y)
        f1 = mean_f1(logits, y)
        crps = mean_crps(logits, y)

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/csi", csi, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/f1", f1, sync_dist=True)
        self.log(f"{stage}/crps", crps, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SaTformerDataModule(pl.LightningDataModule):
    """Lightning DataModule for Sat2Rad dataset."""

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 8,
        num_classes: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Sat2RadDataset(split="train", n_classes=self.hparams.num_classes)
            self.val_dataset = Sat2RadDataset(split="val", n_classes=self.hparams.num_classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )


def main():
    parser = ArgumentParser(description="Train SaTformer with PyTorch Lightning")

    # Training args
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Model args
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--attn", type=str, default="ST^2", choices=["T->S", "S->T", "ST", "ST^2"])
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--ff_dropout", type=float, default=0.1)

    # Logging args
    parser.add_argument("--wandb_project", type=str, default="satformer")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    # Checkpoint args
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Model
    model = SaTformerLightning(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        num_classes=args.num_classes,
        attn=args.attn,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Data
    datamodule = SaTformerDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
    )

    # Logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            log_model=False,
        )
        logger.watch(model, log="gradients", log_freq=100)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="satformer-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
