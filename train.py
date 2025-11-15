import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.simplefilter("ignore", UnsupportedFieldAttributeWarning)

import os
import sys
import yaml
import json
import wandb
import random
import argparse
import importlib


import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Any

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.multiprocessing.set_sharing_strategy("file_system")

from src.util.logger import ExperimentLogger
from src.util.plot.opera import plot_opera_16hr
from src.util.eval_metrics import mean_csi, mean_f1, mean_crps, BinnedEvalMetric, Evaluator
from src.util.scale import scale_zero_to_one, undo_scale_zero_to_one


def ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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
    ddp: False,
    **kwargs
    ) -> DataLoader:
    
    if not ddp:
        dl = DataLoader(dataset, **kwargs)
    else:
        dl = DataLoader(dataset, sampler=DistributedSampler(dataset), **kwargs)
    
    return dl


def train(
    rank            : int,
    config          : dict,
    logger          : ExperimentLogger,
    train_dataloader: DataLoader,
    val_dataloader  : DataLoader,
    model           : Module,
    train_loss      : Module,
    val_loss        : Module,
    optimizer       : Module,
) -> None:

    # use to save model checkpoints
    best_val_loss = float("inf")

    num_epochs    = int(config['global']['num_epochs'])
    device        = rank

    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    train_loss.cuda(rank)
    val_loss.cuda(rank)

    y_label_str      = "y_reg_norm_oh"
    
    # HACK:
    ds: Sat2RadDataset = train_dataloader.dataset
    freqs              = torch.tensor(ds.y_reg_norm_bin_counts)
    freqs              += (1) # no zero weights
    rel_freqs          = 1 / (freqs / freqs.sum())
    
    # not really a PDF; sum is ~1.18
    rel_freqs          = (rel_freqs - rel_freqs.min()) / (rel_freqs.max() - rel_freqs.min())
    weights            = rel_freqs
    b_crps             = BinnedEvalMetric(mean_crps, weights).cuda(rank)
    b_f1               = BinnedEvalMetric(mean_f1, weights).cuda(rank)
    b_csi              = BinnedEvalMetric(mean_csi, weights).cuda(rank)

    # ---------- training loop ----------
    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            X:torch.Tensor = batch["X_norm"].cuda(device)
            y:torch.Tensor = batch[y_label_str].cuda(device)

            # zero gradients
            optimizer.zero_grad()

            # forward; -> [B, N_CLS]
            y_hat:torch.Tensor = model(X)

            loss = train_loss(y_hat, y)
            
            # backprop and step
            loss.backward()
            optimizer.step()

            if config['logging']["wandb"]["log"] == True:
                
                wandb.log(
                    {
                        "epoch"          : epoch,
                        "train_cce_loss" : loss.item(),
                        "train_bw_CSI"   : b_csi(y_hat, y),
                        "train_bw_mF1"   : b_f1(y_hat, y),
                        "train_bw_mCRPS" : b_crps(y_hat, y),
                    }
                )

                save_fig_step = float(config['logging']['wandb']['save_figure_step'])
                
                # if step % save_fig_step == 0:
                #     # log a 16-image mosaic
                #     # TODO: one day... add support for flexible callbacks
                #     y_og = batch["y"]
                #     opera_input_fig = plot_opera_16hr(y_og)
                #     wandb.log({"(y) OPERA": wandb.Image(opera_input_fig)})

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps    = 1

        with torch.no_grad():

            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):

                X:torch.Tensor = batch["X_norm"].cuda(device)
                y:torch.Tensor = batch[y_label_str].cuda(device)

                # predict
                y_hat         = model(X)
                loss          = val_loss(y_hat, y)
                
                # rescaled_loss = val_loss(undo_scale_zero_to_one(y_hat, 0, train_dataset.y_reg_max), batch["y_reg"].cuda(device))
                # note, we still want to use trainset set stats to rescale ^^
                
                if config['logging']["wandb"]["log"] == True:
                    
                    # log some data every step
                    wandb.log(
                        {
                            "epoch"       : epoch,
                            "val_cce_loss": loss.item(),
                            "val_bw_CSI"  : b_csi(y_hat, y),
                            "val_bw_F1"   : b_f1(y_hat, y),
                            "val_bw_CRPS" : b_crps(y_hat, y),
                            }
                    )

                    # save_fig_step = float(config['logging']['wandb']['save_figure_step'])
                    # if step % save_fig_step == 0:
                    #     # log a 16-image mosaic
                    #     y_og = batch["y"]
                    #     opera_input_fig = plot_opera_16hr(y_og)
                    #     wandb.log({"(y) OPERA": wandb.Image(opera_input_fig)})

                # ++
                num_val_steps += 1

            # optional: log best/recent model weights
            avg_val_loss = val_running_loss / num_val_steps

            # HACK: save weights only for 0th process
            if avg_val_loss < best_val_loss and (rank == 0):
                logger.save_weights(model, name="best")
                best_val_loss = avg_val_loss
            elif (rank == 0):
                logger.save_weights(model, name="recent")
                

def main(rank: int, world_size: int, config: dict):

    # init torch.mp process
    ddp_setup(rank, world_size)

    logger = setup_logger(config)
    if config['logging']["wandb"]["log"] == True:

        wandb.login(key=config['logging']["wandb"]["api_key"])
        wandb.init(
            entity="team-levi",
            project="w4c-challenge",
            config=config,
            name=config['logging']['exp_name'],
        )

    model:nn.Module  = create_module(config['model']['target'],              **config['model'  ]['kwargs'])
    train_dataset    = create_module(config['dataset']['train']['target'],   **config['dataset']['train']['kwargs'])
    val_dataset      = create_module(config['dataset']['val'  ]['target'],   **config['dataset']['val'  ]['kwargs'])
    
    train_dataloader = create_dataloader(train_dataset, config['dataloader']['train']['use_ddp'], **config['dataloader']['train']['kwargs'])
    val_dataloader   = create_dataloader(val_dataset,   config['dataloader']['val'  ]['use_ddp'], **config['dataloader']['val'  ]['kwargs'])

    # define loss function and optimizer
    train_loss: torch.nn.Module = create_module(config['loss']['train']['target'], **config['loss']['train']['kwargs'])
    val_loss  : torch.nn.Module = create_module(config['loss']['val'  ]['target'], **config['loss']['val'  ]['kwargs'])

    module_path, class_name = config['optimizer']['target'].rsplit('.', 1)
    module                  = importlib.import_module(module_path)
    _class                  = getattr(module, class_name)
    optimizer:nn.Module     = _class(model.parameters(), **config["optimizer"]["kwargs"])

    # HACK: a pretty terrible way to load model weights from a full .pth model store
    # if config['model']['weights'] != None:
    #     full_model        = torch.load(config['model']['weights'], weights_only=False)
    #     model._parameters = full_model._parameters

    train(rank, config, logger, train_dataloader, val_dataloader, model, train_loss, val_loss, optimizer)
    destroy_process_group()


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
    
    kwargs = dict(args._get_kwargs())
    mp.spawn(main, args=(config['global']['world_size'], config), nprocs=config['global']['world_size'])