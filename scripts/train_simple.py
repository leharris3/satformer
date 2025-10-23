"""
Training script for `sparse-bto` models.
"""

import torch
import yaml
import wandb
import importlib
import argparse
import warnings

from tqdm import tqdm
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from src.util.metrics import PSNR, SSIM
from src.util.logger import Logger
from src.util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS
from src.util.moment_based import cal_moment_based_errs
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import BTOSRDataset, BTO_MANY_RES

warnings.simplefilter("ignore")


def _init_module_from_target(mod_config: dict, *, additional_args: dict={}) -> Any:
    """
    Init a module from a module config dict,
       expect keywords `target` and `args`.
    """
    mod_path, cls_name = mod_config["target"].rsplit(".", 1)
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    args: dict = mod_config.get("args", {})
    args.update(additional_args)
    return cls(**args)


def train(config: dict) -> None:

    logger = _init_module_from_target(config["logger"])

    # some cleaver run initiatization
    if bool(config['wandb']['use_wandb']) == True:
        _init_module_from_target(config['wandb']['login'])
        _init_module_from_target(config['wandb']['init'])

    # init datasets/dataloaders
    train_dataset    = _init_module_from_target(config['train_args']['dataset'])
    val_dataset      = _init_module_from_target(config['val_args']['dataset'])
    train_dataloader = DataLoader(train_dataset, batch_size = int(config['train_args']['batch_size']), shuffle=False)
    val_dataloader   = DataLoader(val_dataset, batch_size = int(config['val_args']['batch_size']), shuffle=False)
    
    # init loss
    train_loss = _init_module_from_target(config['train_args']['loss'])
    val_loss   = _init_module_from_target(config['val_args']['loss'])

    model: torch.nn.Module = _init_module_from_target(config['model'])
    model.float().cuda()

    # init optim
    optimizer: torch.optim.optimizer.Optimizer  = _init_module_from_target(config['train_args']['optimizer'], additional_args={"params": model.parameters()})

    # for weight saving
    best_validation_loss = float("inf")

    # main training loop
    for epoch in range(int(config['train_args']['num_epochs'])):
        
        # train
        model.train()

        for step, item in tqdm(enumerate(train_dataloader), desc=f"🚀 Training Epoch: {epoch + 1}/{int(config['train_args']['num_epochs'])}", total=int(config['train_args']['dataset']['args']['steps_per_epoch'])):

            X       :torch.Tensor = item["X"].float().cuda()
            X_sparse:torch.Tensor = item["X_sparse"].float().cuda()

            # zero gradients
            optimizer.zero_grad()

            # ---- forward: p(y | y_sparse) ----
            X_hat: torch.Tensor = model(X_sparse)

            loss: torch.Tensor = train_loss(X_hat, X)
            loss.backward()
            optimizer.step()

            # ---- log ----

            # calculate moment-based errors
            mb_errs = cal_moment_based_errs(X_hat, X)
            train_mb_errs = {}
            for k in mb_errs:
                train_mb_errs['train_' + k] = mb_errs[k]

            X     = torch.clip(X, 0, 1)
            X_hat = torch.clip(X_hat, 0, 1)

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

            if bool(config['wandb']['use_wandb']) == True:
                log = {
                        "epoch": epoch,
                        "train_l1_loss": loss.item(),
                        "train_psnr": psnr,
                        "train_ssim": ssim,
                    }
                log.update(train_mb_errs)
                wandb.log(log)

            # log figures every 100 steps
            if step % 100 != 0:
                continue

            triplet_name = f"train_epoch_{epoch}_step_{step}.png"

            if isinstance(logger, Logger) and bool(config['wandb']['use_wandb']) == True:
                fig = logger.log_colorized_tensors(
                    (X, "Target (X)"),
                    (X_sparse, "Model Input (X_sparse)"),
                    (X_hat, "Model Prediction"),
                    file_name=triplet_name,
                )
                wandb.log({"Train Qualitative Results": wandb.Image(fig)})

        # validate
        model.eval()
        running_val_loss = 0.

        with torch.no_grad():

            for step, item in tqdm(enumerate(val_dataloader), desc=f"🚀 Validation Epoch: {epoch + 1}/{int(config['train_args']['num_epochs'])}", total=int(config['val_args']['dataset']['args']['steps_per_epoch'])):

                X       :torch.Tensor = item["X"].float().cuda()
                X_sparse:torch.Tensor = item["X_sparse"].float().cuda()
                
                # ---- forward: p(y | y_sparse) ----
                X_hat: torch.Tensor = model(X_sparse)

                loss = val_loss(X_hat, X)

                # calculate moment-based errors
                mb_errs = cal_moment_based_errs(X_hat, X)
                val_mb_errs = {}
                for k in mb_errs:
                    val_mb_errs['val_' + k] = mb_errs[k]
                
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
                        "global_val_step": len(val_dataloader) * (epoch) + step,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": loss.item(),
                    }
                )
                
                if bool(config['wandb']['use_wandb']) == True:
                    log = {
                            "epoch": epoch,
                            "val_l1_loss": loss.item(),
                            "val_psnr": psnr,
                            "val_ssim": ssim,
                        }
                    log.update(val_mb_errs)
                    wandb.log(log)

                # log figures every 100 steps
                if step % 100 != 0:
                    continue

                if isinstance(logger, Logger) and bool(config['wandb']['use_wandb']) == True:
                    triplet_name = f"val_epoch_{epoch}_step_{step}.png"
                    fig = logger.log_colorized_tensors(
                        (X, "Target (X)"),
                        (X_sparse, "Model Input (X_sparse)"),
                        (X_hat, "Model Prediction (X_hat)"),
                        file_name=triplet_name,
                    )
                    wandb.log({"Val Qualitative Results": wandb.Image(fig)})
            
                # accumulate validation loss
                running_val_loss += loss.item()
            
            total_val_steps = int(config['val_args']['dataset']['args']['steps_per_epoch'])
            avg_val_loss = running_val_loss / total_val_steps

            # if best validation perf, save model weights
            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss
                logger.save_weights(model, f"best_epoch_{epoch}")


def main(config: dict) -> None:
    train(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Exeriment run .yaml config.", default="")
    args = parser.parse_args()
    
    assert str(args.config).endswith(".yaml"), f"Error: run config must be a `.yaml` file."
    assert Path(str(args.config)).is_file(),   f"Error: config is not a valid file."
    config_path = Path(str(args.config))

    try:
        with open(str(args.config), "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: exception opening config: {e}")
        raise Exception()

    main(config)
