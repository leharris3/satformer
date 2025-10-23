import os
import sys
import wandb
import random
import argparse
import warnings
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from rich.console import Console
from typing import List, Optional
from torch.utils.data import DataLoader

from src.util.metrics import (
    PSNR,
    SSIM,
    RMSE_surface_roughness_l1,
)
from src.models.unet.unet import UNetSR
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import (
    BTOSRDataset,
    UnifiedMOS2SRDataset,
    MOS2SRDataset,
    MOS2_SEF_FULL_RES_SRC_DIR,
    MOS2_SILICON_DIR,
    MOS2_SAPPHIRE_DIR,
    MOS2_SYNTHETIC,
    BTO_MANY_RES,
)
from src.util.logger import ExperimentLogger
from src.util.config import (
    TrainConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    MODELS,
)
from src.util.loss import roughness_loss, rotation_invariant_l1_loss

warnings.simplefilter("always")
torch.multiprocessing.set_sharing_strategy("file_system")
TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
CONSOLE = Console()


def setup_logger(
    train_config: TrainConfig, model_config: Optional[ModelConfig]
) -> ExperimentLogger:

    logger = ExperimentLogger(
        train_config_dict=train_config.to_dict(),
        model_config_dict=model_config.to_dict() if model_config != None else None,
        root=train_config.log_root,
        exp_name=train_config.exp_name,
        log_interval=train_config.log_interval,
    )
    logger.add_result_columns(train_config.result_columns)
    return logger


def create_model(config: TrainConfig) -> nn.Module:
    model_fn = MODELS[config.model_name]["fn"]
    model_weights = MODELS[config.model_name]["weights"]
    if model_weights:
        model = model_fn(weights=model_weights)
    elif config.model_name == "hiera":
        model = model_fn
        model.freeze()
    else:
        model = model_fn()
    assert isinstance(model, nn.Module)
    return model


def create_dataloader(args, config: TrainConfig, split: str) -> DataLoader:

    assert str(args.dataset) in [
        "all",
        "synthetic",
        "bto",
        "mos2-sef",
        "sapphire",
        "silicon",
    ]

    src_dir = {
        "all": None,
        "synthetic": MOS2_SYNTHETIC,
        "mos2-sef": MOS2_SEF_FULL_RES_SRC_DIR,
        "sapphire": MOS2_SAPPHIRE_DIR,
        "silicon": MOS2_SILICON_DIR,
        "bto": BTO_MANY_RES,
    }[args.dataset]

    dataset = None
    if str(args.dataset) == "all":
        dataset = UnifiedMOS2SRDataset(
            split=split,
            steps_per_epoch=(
                int(config.steps_per_epoch * config.train_batch_size)
                if split == "train"
                else config.val_steps_per_epoch
            ),
            upsample_factor=int(args.upsample_factor),
        )
    elif str(args.dataset) == "bto":
        dataset = BTOSRDataset(
            steps_per_epoch=(
                int(config.steps_per_epoch * config.train_batch_size)
                if split == "train"
                else config.val_steps_per_epoch
            ),
            upsample_factor=int(args.upsample_factor),
        )
    else:
        dataset = MOS2SRDataset(
            src_dir=src_dir,
            split=split,
            steps_per_epoch=(
                int(config.steps_per_epoch * config.train_batch_size)
                if split == "train"
                else config.val_steps_per_epoch
            ),
            upsample_factor=int(args.upsample_factor),
        )

    return DataLoader(
        dataset,
        batch_size=(
            config.train_batch_size if split == "train" else config.val_batch_size
        ),
        shuffle=False,
        num_workers=config.num_workers,
    )


def train(
    args,
    config: TrainConfig,
    model_config: Optional[ModelConfig] = None,
) -> None:

    logger = setup_logger(config, model_config)

    # wandb login
    wandb.login(key="3d8c09b359c1abc995fd03c27398c41afce857c1")
    wandb.init(
        entity="team-levi",
        project="sparse-cafm",
        config=config.to_dict(),
        name=str(args.exp_name),
    )

    # HACK: just loading a torch .pth file
    # model = create_model(config)
    # model = SwinCAFM.init_from_config(model_config.to_dict())
    model = torch.load(str(args.weights))

    train_dataloader = create_dataloader(args, config, "train")
    val_dataloader = create_dataloader(args, config, "val")

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config.train_loss]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config.val_loss]()

    # use to save model checkpoints
    best_val_loss = float("inf")

    num_epochs = config.epochs
    device = config.device

    # as per: https://arxiv.org/pdf/2404.00722
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    # assert isinstance(model, SwinCAFM)
    # HACK: randomly init weights
    # model.apply(model._init_weights)

    model.cuda(device)
    model.float()

    # ---------- training loop ----------
    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            # [0, 1]
            # NOTE: manually specifing X vs y
            X = batch["X"].float().cuda()
            X_sparse = batch["X_sparse"].float().cuda()

            # zero gradients
            optimizer.zero_grad()

            # ---- forward: p(y | y_sparse) ----
            X_hat: torch.Tensor = model(X_sparse)

            assert isinstance(train_dataloader.dataset, BTOSRDataset)
            rmse_sr_loss = RMSE_surface_roughness_l1(
                X,
                X_hat,
                train_dataloader.dataset.topo_maps_min,
                train_dataloader.dataset.topo_maps_max,
            )

            # --- L1 ----
            # loss = torch.nn.functional.l1_loss(X, X_hat)

            # --- L1 + surface_roughness ----
            # EPS = 1.5
            # loss = torch.nn.functional.l1_loss(X, X_hat) + (EPS * rmse_sr_loss)

            # --- surface_roughness ---
            loss = rotation_invariant_l1_loss(
                model,
                X,
                X_sparse,
                train_dataloader.dataset.topo_maps_min,
                train_dataloader.dataset.topo_maps_max,
            )

            # backprop and step
            loss.backward()
            optimizer.step()

            # HACK: clip to [0, 1]
            X     = torch.clip(X, 0, 1)
            X_hat = torch.clip(X_hat, 0, 1)

            # ---- add dummy dims for PSNR/SSIM ----
            X_il    : torch.Tensor = X.unsqueeze(1).repeat(1, 3, 1, 1)
            X_hat_il: torch.Tensor = X_hat.unsqueeze(1).repeat(1, 3, 1, 1)

            psnr = PSNR(X_il, X_hat_il, (0, 1))
            ssim = SSIM(X_il, X_hat_il, (0, 1))

            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + step,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": None,
                }
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_l1_loss": loss.item(),
                    "train_psnr": psnr,
                    "train_ssim": ssim,
                    "train_RMSE_surface_roughness_l1": rmse_sr_loss,
                }
            )

            # log figures every 100 steps
            if step % 100 != 0:
                continue

            triplet_name = f"train_epoch_{epoch}_step_{step}.png"
            fig = logger.log_colorized_tensors(
                (X, "Target (X)"),
                (X_sparse, "Model Input (X_sparse)"),
                (X_hat, "Model Prediction"),
                file_name=triplet_name,
            )

            wandb.log({"Train Qualitative Results": wandb.Image(fig)})

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps = 1

        with torch.no_grad():

            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):

                # NOTE: manually specifing X vs y
                X        = batch["X"].float().cuda()
                X_sparse = batch["X_sparse"].float().cuda()

                # ---- forward: p(y | y_sparse) ----
                X_hat: torch.Tensor = model(X_sparse)

                assert isinstance(train_dataloader.dataset, BTOSRDataset)
                rmse_sr_loss = RMSE_surface_roughness_l1(
                    X,
                    X_hat,
                    train_dataloader.dataset.topo_maps_min,
                    train_dataloader.dataset.topo_maps_max,
                )

                # --- L1 ----
                # loss = val_loss(X_hat, X)

                # --- Surface Roughness ---

                loss = roughness_loss(
                    X_hat,
                    X,
                    train_dataloader.dataset.topo_maps_min,
                    train_dataloader.dataset.topo_maps_max,
                )

                val_running_loss += loss.item() * X.size(0)

                X = torch.clip(X, 0, 1)
                X_hat = torch.clip(X_hat, 0, 1)

                # ---- add dummy dims for PSNR/SSIM ----
                X_il: torch.Tensor = X.unsqueeze(1).repeat(1, 3, 1, 1)
                X_hat_il: torch.Tensor = X_hat.unsqueeze(1).repeat(1, 3, 1, 1)

                psnr = PSNR(X_il, X_hat_il, (0, 1))
                ssim = SSIM(X_il, X_hat_il, (0, 1))

                logger.log(
                    **{
                        "global_train_step": None,
                        "global_val_step": len(val_dataloader) * (epoch) + i,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": loss.item(),
                    }
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "val_l1_loss": loss.item(),
                        "val_psnr": psnr,
                        "val_ssim": ssim,
                        "val_RMSE_surface_roughness_l1": rmse_sr_loss,
                    }
                )

                # log figures every 100 steps
                if i % 100 != 0:
                    continue

                triplet_name = f"val_epoch_{epoch}_step_{i}.png"

                fig = logger.log_colorized_tensors(
                    (X, "Target (X)"),
                    (X_sparse, "Model Input (X_sparse)"),
                    (X_hat, "Model Prediction (X_hat)"),
                    file_name=triplet_name,
                )
                wandb.log({"Val Qualitative Results": wandb.Image(fig)})

                # ++
                num_val_steps += 1

            # optional: log best/recent model weights
            avg_val_loss = val_running_loss / num_val_steps

            if not bool(config.save_weights):
                continue
            if bool(config.save_only_best_weights):
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logger.save_weights(model, "best")
                else:
                    # NOTE: we overwrite previous "latest" weights
                    logger.save_weights(model, f"latest")
            else:
                logger.save_weights(model, f"epoch_{epoch}")


def main(args: argparse.Namespace) -> None:

    # load training config
    config = TrainConfig(TRAIN_CONFIG_FP)
    config.weights = args.weights

    model_config: Optional[ModelConfig] = None

    # optional: parse model config
    if config.model_config_file != None:
        model_config_abs_path = os.path.join(
            Path(TRAIN_CONFIG_FP).parent.__str__(), config.model_config_file
        )
        assert os.path.isfile(
            model_config_abs_path
        ), f"Bad path to model config: {model_config_abs_path}"
        model_config = ModelConfig(model_config_abs_path)

    # -------------------- training config args --------------------
    config.exp_name = args.exp_name
    config.log_root = args.root
    # config.learning_rate = str(args.learning_rate)
    # config.train_batch_size = int(args.batch_size)
    # -------------------- model config args --------------------
    if model_config != None:
        # transformer block depths; e.g., [6, 6, 6, 6, 6, 6]
        model_config.depths = [args.depths] * args.num_blocks
        # num heads per block; e.g., [6, 6, 6, 6, 6, 6]
        model_config.num_heads = [args.num_heads] * args.num_blocks
        # size of sifted-attention window
        model_config.window_size = args.window_size
        model_config.drop_path_rate = args.drop_path_rate
        model_config.norm_layer = args.norm_layer

    args.upsample_factor = int(args.upsample_factor)

    # train
    train(args, config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------- training config args --------------------
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="Experiment directory name.",
        default="my-experiment",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        help="Root directory to save experiment in.",
        default="__exps__/",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        help="'synthetic', 'mos2-sef', 'sapphire', 'silicon', 'all']",
        default="mos2-sef",
    )
    parser.add_argument(
        "-ws", "--weights", type=str, help="Path to model checkpoints", default=""
    )
    parser.add_argument(
        "-fm", "--formulation", type=str, help="['X', 'y', 'both']", default="y"
    )
    # -------------------- model config args --------------------
    parser.add_argument(
        "-dps", "--depths", type=int, help="Depths of RSTB blocks", default=6
    )
    parser.add_argument(
        "-nbs", "--num_blocks", type=int, help="Number of RSTB blocks", default=6
    )
    parser.add_argument(
        "-nhs",
        "--num_heads",
        type=int,
        help="Number of heads per RSTB block",
        default=6,
    )
    parser.add_argument(
        "-wsz",
        "--window_size",
        type=int,
        help="Size of shifted attention window",
        default=8,
    )
    parser.add_argument("-dpr", "--drop_path_rate", type=float, help="", default=0.1)
    parser.add_argument(
        "-nlr", "--norm_layer", type=str, help="", default="torch.nn.LayerNorm"
    )
    # -------------------- ablation args --------------------
    parser.add_argument("-sw", "--surrogate_weights", type=str, help="", default="")
    parser.add_argument("-lr", "--learning_rate", type=float, help="", default=1e-5)
    parser.add_argument("-bs", "--batch_size", type=int, help="", default=1)
    parser.add_argument("-sr", "--upsample_factor", type=int, help="", default=2)
    args = parser.parse_args()

    main(args)
