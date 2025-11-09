import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.simplefilter("ignore", UnsupportedFieldAttributeWarning)

import os
import importlib
import argparse
import warnings
import yaml
import json
import subprocess
import pandas as pd
import torch.nn.functional as F

import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from src.util.logger import ExperimentLogger
from src.dataloader.challenge_one_dataloader import Sat2RadDataset
from src.dataloader.dataset_stats import Y_REG_NORM_BINS, Y_REG_MAX

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


SUBMISSION_FOLDER_SUBDIR = "submission-bins-1-all-regions"


def create_module(target: str, **kwargs):
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

    model  :nn.Module      = create_module(config['model']['target'],           **config['model']['kwargs'])
    dataset:Sat2RadDataset = create_module(config['dataset']['test']['target'], **config['dataset']['test']['kwargs'])
    dataloader             = create_dataloader(dataset,                         **config['dataloader']['test']['kwargs'])
    device                 = int(config['global']['device'])
    
    weights_fp        = config['model']['weights']

    try:

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        wrapper = torch.load(weights_fp, map_location='cpu', weights_only=False)
        # model = wrapper.module.cpu()
        state_dict = wrapper.module.state_dict()
    
    finally:
        
        torch.distributed.destroy_process_group()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    preds = {}

    for _, batch in enumerate(
        tqdm(dataloader, total=len(dataset))
    ):

        X    : torch.Tensor = batch["X_norm_32"].to(device)
        
        # [B, 128]
        y_hat:torch.Tensor  = model(X)
        
        # get CDF of model predictions
        logits              = F.softmax(y_hat, dim=1)
        cum_prob            = logits.cumsum(dim=1)

        # [0, 1] -> [D.min, D.max]
        rescaled_bins       = (dataset.y_reg_norm_bins.cuda(device) * Y_REG_MAX).repeat(cum_prob.shape[0], 1)

        csv_fp = submission_dir / Path(f"{batch["year"].item()}") / Path(f"{batch['file_name'][0].split(".")[0]}.test.cum4h.csv")
        if csv_fp not in preds: preds[csv_fp] = []

        # HACK:
        # [Case-ID, amount (mm/hr), cum_prob]
        for B in range(cum_prob.shape[0]):
            rb, cp = rescaled_bins[B, ...], cum_prob[B, ...]            

            for _bin, _prob in zip(rb, cp):
                
                # HACK: examples show int bins
                # * we can do floats, but score drops ):
                # _bin = int(_bin.item())
                _bin = int(_bin.item())

                # HACK: floor and ceil
                if _bin > 1000 :
                    preds[csv_fp].append([batch['Case-id'][0], _bin, 1.0])
                elif _bin < 0.0:
                    preds[csv_fp].append([batch['Case-id'][0], _bin, 0.0])
                else:
                    preds[csv_fp].append([batch['Case-id'][0], _bin, _prob.item()])

    # save all predictions as csvs
    for k, v in preds.items():
        df = pd.DataFrame(v)
        df.to_csv(k, index=False, header=False)

    # cd into exp dir
    os.chdir(str(submission_dir))

    # store as zip
    command = ["zip", "-r", "../submission-bins-1-all-regions.zip", ".", "i", "*", "&", "exit"]
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
