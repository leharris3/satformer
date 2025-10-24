import os
import sys
import wandb
import random
import importlib
import argparse
import warnings
import yaml
import json

import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader

from src.util.logger import ExperimentLogger

warnings.simplefilter("always")
torch.multiprocessing.set_sharing_strategy("file_system")


def create_module(target: str, **kwargs):
    """
    Args
    ---
    :target: module path to class (e.g., `src.model.toy.ToyCummulativePrecipitationModel`)
    """

    module_path, class_name = target.rsplit('.', 1)
    module                  = importlib.import_module(module_path)
    _class                  = getattr(module, class_name)
    instance                = _class(**kwargs)
    
    return instance


def setup_logger(
    config: str,
) -> ExperimentLogger:

    logger = ExperimentLogger(
        train_config_dict=config,
        root=config['logging']['root'],
        exp_name=config['logging']['exp_name'],
        log_interval=int(config['logging']['log_interval']),
    )

    # TODO: figure out what we we're doing here
    # logger.add_result_columns(train_config.result_columns)
    return logger


def create_dataloader(
    dataset: torch.utils.data.Dataset, 
    **kwargs
    ) -> DataLoader:
    return DataLoader(
        dataset,
        **kwargs,
    )


def train(
    config: dict,
    **kwargs
) -> None:

    logger = setup_logger(config)

    if config['logging']["wandb"]["log"] == True:

        wandb.login(key=config['logging']["wandb"]["api_key"])
        wandb.init(
            entity="team-levi",
            project="w4c-challenge",
            config=config,
            name=config['logging']['exp_name'],
        )

    model:nn.Module  = create_module(config['model']['target'],            **config['model']['kwargs'])
    train_dataset    = create_module(config['dataset']['train']['target'], **config['dataset']['train']['kwargs'])
    val_dataset      = create_module(config['dataset']['val']['target']  , **config['dataset']['val']['kwargs'])
    
    train_dataloader = create_dataloader(train_dataset, **config['dataloader']['train']['kwargs'])
    val_dataloader   = create_dataloader(val_dataset  , **config['dataloader']['val']['kwargs'])

    # define loss function and optimizer
    train_loss: torch.nn.Module = create_module(config['loss']['train']['target'], **config['loss']['train']['kwargs'])
    val_loss  : torch.nn.Module = create_module(config['loss']['val']['target'],   **config['loss']['val']['kwargs'])

    # use to save model checkpoints
    best_val_loss = float("inf")
    num_epochs    = int(config['global']['num_epochs'])
    device        = int(config['global']['device'])

    module_path, class_name = config['optimizer']['target'].rsplit('.', 1)
    module                  = importlib.import_module(module_path)
    _class                  = getattr(module, class_name)
    optimizer:nn.Module     = _class(model.parameters(), **config["optimizer"]["kwargs"])

    model.cuda(device)
    # model.float()

    # ---------- training loop ----------
    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            X:torch.Tensor = batch["X"].cuda()
            y:torch.Tensor = batch["y"].cuda()

            # zero gradients
            optimizer.zero_grad()

            # predict
            y_hat = model(X)

            loss = train_loss(y_hat, y)
            
            # backprop and step
            loss.backward()
            optimizer.step()

            # logger.log(
            #     **{
            #         "global_train_step": len(train_dataloader) * (epoch) + step,
            #         "global_val_step": None,
            #         "epoch": epoch,
            #         "train_loss": loss.item(),
            #         "val_loss": None,
            #     }
            # )
            # wandb.log(
            #     {
            #         "epoch": epoch,
            #         "train_l1_loss": loss.item(),
            #         "train_psnr": psnr,
            #         "train_ssim": ssim,
            #         "train_RMSE_surface_roughness_l1": rmse_sr_loss,
            #     }
            # )

            # # log figures every 100 steps
            # if step % 100 != 0:
            #     continue

            # ... figure logging logic
            # wandb.log({"Train Qualitative Results": wandb.Image(fig)})

            pass

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps    = 1

        with torch.no_grad():

            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):

                # NOTE: manually specifing X vs y
                # # ... load X, y
                
                # ... loss
                # loss = val_loss(X_hat, X)

                # logger.log(
                #     **{
                #         "global_train_step": None,
                #         "global_val_step": len(val_dataloader) * (epoch) + i,
                #         "epoch": epoch,
                #         "train_loss": None,
                #         "val_loss": loss.item(),
                #     }
                # )
                # wandb.log(
                #     {
                #         "epoch": epoch,
                #         "val_l1_loss": loss.item(),
                #         "val_psnr": psnr,
                #         "val_ssim": ssim,
                #         "val_RMSE_surface_roughness_l1": rmse_sr_loss,
                #     }
                # )

                # # log figures every 100 steps
                # if i % 100 != 0:
                #     continue

                # ... figure logging logic
                # wandb.log({"Val Qualitative Results": wandb.Image(fig)})

                # # ++
                # num_val_steps += 1

                pass

            # optional: log best/recent model weights
            avg_val_loss = val_running_loss / num_val_steps

            # ... handle optional ckpt saving


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # -------------------- training config args --------------------
    parser.add_argument(
        "-c",
        "--config_fp",
        type=str,
        help="Path to experiment `.json` config file.",
        default="",
    )

    args:argparse.Namespace   = parser.parse_args()
    with open(args.config_fp, 'r') as f:
        config = json.load(f)
    
    train(config, **dict(args._get_kwargs()))
