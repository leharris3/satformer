import os
import sys
import wandb
import random
import importlib
import argparse
import warnings
import yaml
import json
import subprocess
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Any
from torch.utils.data import DataLoader

from src.util.logger import ExperimentLogger
from src.util.plot.opera import plot_opera_16hr
from src.util.scale import scale_zero_to_one, undo_scale_zero_to_one

warnings.simplefilter("always")
torch.multiprocessing.set_sharing_strategy("file_system")


SUBMISSION_FOLDER_SUBDIR = "submission-bins-1-all-regions"


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


@torch.no_grad()
def test(
    config: dict,
    **kwargs
) -> None:

    # --- set up experiment dirs

    logger         = setup_logger(config)
    exp_dir        = logger.exp_dir
    
    submission_dir = Path(exp_dir) / Path(SUBMISSION_FOLDER_SUBDIR)
    os.makedirs(submission_dir)

    sub_dir_2019   = Path(submission_dir) / Path("2019")
    sub_dir_2020   = Path(submission_dir) / Path("2020")

    os.makedirs(sub_dir_2019)
    os.makedirs(sub_dir_2020)

    if config['logging']["wandb"]["log"] == True:

        wandb.login(key=config['logging']["wandb"]["api_key"])
        wandb.init(
            entity="team-levi",
            project="w4c-challenge",
            config=config,
            name=config['logging']['exp_name'],
        )

    model:nn.Module  = create_module(config['model']['target'],           **config['model']['kwargs'])
    dataset          = create_module(config['dataset']['test']['target'], **config['dataset']['test']['kwargs'])
    dataloader       = create_dataloader(dataset,                         **config['dataloader']['test']['kwargs'])

    device = int(config['global']['device'])
    
    # HACK:
    model_fp = "/playpen-ssd/levi/w4c/w4c-25/__exps__/2025-11-01_18-11-20_timesformer-reg-bs=1-loss=l1/best.pth"
    model    = torch.load(model_fp, weights_only=False)

    model.cuda(device)
    model.eval()

    preds = {}

    for step, batch in enumerate(
        tqdm(dataloader, total=len(dataset))
    ):

        X: torch.Tensor = batch["X_norm"].cuda()

        # forward; [B, C=1, H, W]
        try:
            y_hat:torch.Tensor = model(X)
        except: breakpoint()
        
        y_hat_scaled       = undo_scale_zero_to_one(y_hat, 0, dataset.y_reg_max)

        # HACK
        if not (y_hat_scaled > 0):
            y_hat_scaled = torch.Tensor([0])

        csv_fp = submission_dir / Path(f"{batch["year"].item()}") / Path(f"{batch['file_name'][0].split(".")[0]}.test.cum4h.csv")
        if csv_fp not in preds: preds[csv_fp] = []

        # HACK:
        # [Case-ID, amount (mm/hr), cum_prob]
        preds[csv_fp].append([batch['Case-id'][0], y_hat_scaled.item(), 1])

        # if config['logging']["wandb"]["log"] == True:
            
        #     wandb.log(
        #         {
        #             "train_loss": loss.item(),
        #         }
        #     )

    # save all predictions as csvs
    for k, v in preds.items():
        df = pd.DataFrame(v)
        df.to_csv(k, index=False, header=False)

    # cd into exp dir
    os.chdir(str(submission_dir))

    # store as zip
    command = ["zip", "-r", "../submission-bins-1-all-regions.zip", ".", "i", "*"]
    subprocess.run(command)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # -------------------- test bench config args --------------------
    parser.add_argument(
        "-c",
        "--config_fp",
        type=str,
        help="Path to experiment `.json` config file.",
        default="",
    )

    args: argparse.Namespace = parser.parse_args()
    with open(args.config_fp, 'r') as f:
        config = json.load(f)
    
    test(config, **dict(args._get_kwargs()))
