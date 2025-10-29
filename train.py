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
from typing import List, Optional, Any
from torch.utils.data import DataLoader

from src.util.logger import ExperimentLogger
from src.util.plot.opera import plot_opera_16hr

warnings.simplefilter("always")
torch.multiprocessing.set_sharing_strategy("file_system")


def create_module(target: str, **kwargs) -> Any:
    """
    Args
    ---
    :target: module path to class (e.g., `src.model.toy.ToyCummulativePrecipitationModel`)
    
    Return
    ---
    :Any:
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

            # [B, T=4, C=11, H, W] ->
            # [B, C=11, H, W]
            X = X.mean(dim=1)

            # [B, T=16, C=1, H, W] ->
            # [B, T=16, H, W]      ->
            # [B, H, W]            ->
            # [B, 1]
            y_regression_target = (y.squeeze(2).mean(dim=1) * 4).mean(dim=(1, 2)).unsqueeze(1)

            # zero gradients
            optimizer.zero_grad()

            # predict
            # [B, C=1, H, W]
            y_hat:torch.Tensor      = model(X)

            # HACK
            y_hat_regression_pred = y_hat.squeeze(1).mean(dim=(1, 2)).unsqueeze(1)

            # loss = train_loss(y_hat, y)

            # MSE regression loss
            loss = train_loss(y_hat_regression_pred, y_regression_target)
            breakpoint()
            
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

            if config['logging']["wandb"]["log"] == True:
                
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": loss.item(),
                    }
                )

                save_fig_step = float(config['logging']['wandb']['save_figure_step'])
                if step % save_fig_step == 0:
                    # log a 16-image mosaic
                    # TODO: one day... add support for flexible callbacks
                    opera_input_fig = plot_opera_16hr(y)
                    wandb.log({"(y) OPERA": wandb.Image(opera_input_fig)})

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps    = 1

        with torch.no_grad():

            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):

                X:torch.Tensor = batch["X"].cuda()
                y:torch.Tensor = batch["y"].cuda()

                # [B, T=4, C=11, H, W] ->
                # [B, C=11, H, W]
                X = X.mean(dim=1)

                # [B, T=16, C=1, H, W] ->
                # [B, T=16, H, W]      ->
                # [B, H, W]            ->
                # [B, 1]
                y_regression_target = (y.squeeze(2).mean(dim=1) * 4).mean(dim=(1, 2)).unsqueeze(1)

                # zero gradients
                optimizer.zero_grad()

                # predict
                y_hat = model(X)

                # HACK
                y_hat_regression_pred = y_hat.squeeze(1).mean(dim=(1, 2)).unsqueeze(1)

                # loss = val_loss(y_hat, y)

                # MSE regression loss
                loss = val_loss(y_hat_regression_pred, y_regression_target)

                # logger.log(
                #     **{
                #         "global_train_step": None,
                #         "global_val_step": len(val_dataloader) * (epoch) + i,
                #         "epoch": epoch,
                #         "train_loss": None,
                #         "val_loss": loss.item(),
                #     }
                # )
                
                if config['logging']["wandb"]["log"] == True:
                    
                    # log some data every step
                    wandb.log(
                        {
                            "epoch": epoch,
                            "val_loss": loss.item(),
                        }
                    )

                    save_fig_step = float(config['logging']['wandb']['save_figure_step'])
                    if step % save_fig_step == 0:
                        # log a 16-image mosaic
                        # TODO: one day... add support for flexible callbacks
                        opera_input_fig = plot_opera_16hr(y)
                        wandb.log({"(y) OPERA": wandb.Image(opera_input_fig)})

                # # ++
                # num_val_steps += 1

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
