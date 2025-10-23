import os
import math
import torch
import datetime
import yaml
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.util.torch_helpers import convert_to_img_like
from src.util.config import parse_config

EXPS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/"
FIGURES_DIR_NAME = "figures"
RESULTS_CSV_NAME = "results.csv"


class Logger:
    """
    A slightly more flexible logger that doesn't require config files.
    The user must call ._flush to write.
    """

    def __init__(self, root: str, exp_name: str):
        """
        :param root: path to dir to log experiment
        """

        # path to experiment
        assert Path(root).is_dir(), f"Error: not a valid dir: {root}"
        self.root = root
        
        try:
            os.makedirs(self.root, exist_ok=True)
        except:
            raise Exception(
                f"Could not create a new experiment directory @: \n \
                {self.root}"
            )

        # name of new subdir for expeiment
        self.exp_name = exp_name

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now = now.replace(" ", "_").replace(":", "-")

        self.exp_dir: Path = Path(root) / Path(now + "_" + exp_name)

        try:
            os.makedirs(str(self.exp_dir), exist_ok=True)
        except:
            raise Exception(
                f"Could not create a new experiment directory @: \n \
                {os.path.join(self.root, self.exp_name)}"
            )

        self.results_out_path = os.path.join(str(self.exp_dir), "results.csv")

        # logs
        self.results = pd.DataFrame()
        self.log_buffer = []

    def _flush(self):

        if not self.log_buffer:
            return

        # init new results table from buffer
        _logs = pd.DataFrame.from_records(self.log_buffer)

        # append results in memory
        self.results = pd.concat([self.results, _logs], ignore_index=True)
        if not os.path.exists(self.results_out_path):
            # create new file
            _logs.to_csv(self.results_out_path, index=False)
        else:
            # write to csv in append mode
            _logs.to_csv(self.results_out_path, mode="a", header=False, index=False)

        self.log_buffer = []

    def log(self, **kwargs) -> None:

        # append results to mem
        self.log_buffer.append(kwargs)
        self._flush()

    def log_colorized_tensors(
        self, *samples: Tuple[torch.Tensor, str], file_name: str
    ) -> plt.Figure:
        """
        Log tensors with the exact shape: [B, H, W], using an added color pallet to make things pretty.
        """

        MAX_COLS = 3
        IMAGE_SIZE_IN = 6
        num_images = len(samples)
        n_cols = min(num_images, MAX_COLS)
        n_rows = math.ceil(num_images / MAX_COLS)

        # TODO: is 4-inches enough?... (;
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * IMAGE_SIZE_IN, n_rows * IMAGE_SIZE_IN)
        )

        # axes always 2d arr
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for idx, (tensor, name) in enumerate(samples):
            row = idx // MAX_COLS
            col = idx % MAX_COLS
            # only use first tensor in batch
            img = tensor[0, ...]
            # strange, convert to img like returns a list...
            img = convert_to_img_like(img)[0]
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(name, fontsize=14)
            ax.axis("off")

        # turn off extra subplots
        # idk, chat thinks this a good idea
        total_cells = n_rows * n_cols
        for idx in range(num_images, total_cells):
            row = idx // MAX_COLS
            col = idx % MAX_COLS
            axes[row, col].axis("off")

        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, file_name)
        plt.savefig(out_fp, bbox_inches="tight", pad_inches=0.1, dpi=300)
        return fig

    def save_weights(
        self,
        model: torch.nn.Module,
        name: str = "best",
    ) -> None:
        """
        Save model weights of a `torch.nn.Module` object to the current exp dir.

        :param model: model to save
        """

        out_fp = Path(self.exp_dir) / Path(f"{name}.pth")
        torch.save(model, str(out_fp))


class ExperimentLogger:
    """
    A flexible logger used to record and organize experimental runs.
    """

    def __init__(
        self,
        train_config_dict: dict,
        model_config_dict: Optional[dict] = None,
        root: str = EXPS_DIR,
        exp_name: Optional[str] = "",
        log_interval: int = 100,
        enable_tensorboard=False,
        enable_wandb=False,
        wandb_proj_name: Optional[str] = None,
    ) -> None:
        """
        :param config_fp:           path to a `.yaml` config file containing all hps
        :param root:                path to top experiment dir
        :param exp_name:            name of the experiment
        :param log_interval:        how often to write log results to .csv file
        :param enable_tensorboard:  flag to enable tensorboard logging
        :param enable_wandb:        flag to enable W&B logging [NOT SUPPORTED]
        :param wandb_project_name:  name of W&B project (e.g. "my-project")
        """

        self.config: dict = train_config_dict
        self.model_config: Optional[dict] = model_config_dict
        self.exp_name: str = exp_name

        self.results = pd.DataFrame()
        self.log_buffer = []
        self.log_interval: int = log_interval
        self.log_counter = 0

        self.root: str = root
        self.exp_dir: Optional[str] = None

        # ---- tensorboard support ----
        self.enable_tensorboard: bool = enable_tensorboard
        self.results_out_path: Optional[str] = None
        self.summary_writer: Optional[SummaryWriter] = None

        # ---- wandb support ----
        self.enable_wandb = enable_wandb
        if self.enable_wandb == True:
            assert (
                wandb_proj_name != None
            ), f"Error: must provide a valid name for wandb_proj_name"
        self.wandb_proj_name = wandb_proj_name
        self.wandb_run = None

        self._setup_exp_dir()

    def _flush(self) -> None:
        if not self.log_buffer:
            return
        # init new results table from buffer
        _logs = pd.DataFrame.from_records(self.log_buffer)

        # append results in memory
        self.results = pd.concat([self.results, _logs], ignore_index=True)
        if not os.path.exists(self.results_out_path):
            # create new file
            _logs.to_csv(self.results_out_path, index=False)
        else:
            # write to csv in append mode
            _logs.to_csv(self.results_out_path, mode="a", header=False, index=False)
        self.log_buffer = []

    def _update_csv(self) -> None:
        self.results.to_csv(self.results_out_path, index=False)

    def _setup_exp_dir(self) -> None:

        # get date and time as a string
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name = date_time_str + "_" + self.exp_name
        exp_out_dir = os.path.join(self.root, subdir_name)
        self.exp_dir = exp_out_dir

        # make new subdir if needed
        os.makedirs(exp_out_dir, exist_ok=True)

        # save config in subdir
        config_save_fp = os.path.join(exp_out_dir, "config.yaml")
        with open(config_save_fp, "w") as f:
            yaml.dump(self.config, f, indent=4)

        # path to results csv file
        self.results_out_path = os.path.join(exp_out_dir, RESULTS_CSV_NAME)

        # optional: create a tensorboard writer object
        if self.enable_tensorboard:
            tb_log_dir = os.path.join(self.exp_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=tb_log_dir)

        # optional: create a wandb run
        if self.enable_wandb:
            with open(config_save_fp, "r") as f:
                config_dict = yaml.safe_load(f)
            wandb.init(
                project=self.wandb_proj_name,
                name=self.exp_name,
                config=config_dict,
                dir=self.exp_dir,
            )
            self.wandb_run = wandb.run

        model_config_save_fp = os.path.join(exp_out_dir, "model.yaml")

        # save a copy of the model config to the exp dir
        with open(model_config_save_fp, "w") as f:
            yaml.dump(self.model_config, f, indent=4)

        # TODO: this looks hacky; remove
        self.config_fp = config_save_fp

    def add_result_column(self, name: str) -> None:
        self.results[name] = None
        # HACK: just ignore this for now
        # self._update_csv()

    def add_result_columns(self, names: List[str]) -> None:
        for name in names:
            self.add_result_column(name)
        # HACK: just ignore this for now
        # self._update_csv()

    def log(self, **kwargs) -> None:
        """
        Log a dictionary of items to a csv.
        """

        # append results to mem
        self.log_buffer.append(kwargs)
        self.log_counter += 1

        # write to out
        if len(self.log_buffer) >= self.log_interval:
            self._flush()

        # optional: log -> tensorboard
        if self.enable_tensorboard:
            if step is None:
                step = self.log_counter
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self.summary_writer.add_scalar(k, v, step)

        # optional: log -> wandb
        if self.enable_wandb:
            step = self.log_counter
            wandb_dict = {
                k: v for k, v in kwargs.items() if isinstance(v, (int, float))
            }
            wandb.log(wandb_dict, step=step)

    def save_weights(
        self,
        x: Union[torch.nn.Module, pytorch_lightning.trainer.Trainer],
        name: str = "best",
    ) -> None:
        """
        TODO: support `torch.nn.Module`

        Save model weights of a `torch.nn.Module` object to the current exp dir.

        :param x: model to save
        """

        # TODO:
        # for some reason we can load ControlNet models from the first ckpt
        # but not from subsequent saves.
        # also, model weights appear to grow in size over training run, implying that we are saving some
        # info that we shouldn't (e.g., logs).

        # NOTE:
        # 1. increased model size does not seem to be related to use appending to an existing file.
        # 2. we CAN load weights from subsequent saves with DIFFERENT names.
        # 3. we CAN load weights from subsequent saves with IDENTICAL names.
        # 4. can only conclude that the file suffix was the issue lol

        model_out_path = os.path.join(self.exp_dir, f"{self.exp_name}_{name}.pth")
        if isinstance(x, pytorch_lightning.trainer.Trainer):
            x.save_checkpoint(model_out_path.replace(".pth", ".ckpt"))
        else:
            torch.save(x, model_out_path)

        # if pickle_weights == True:
        #     with open(model_out_path.replace(".pth", ".pkl"), 'wb') as f:
        #         pickle.dump(x, f)
        # else:
        #     torch.save(x, model_out_path)

    def save_tensorlike_data(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray],
        subdir: Optional[str] = None,
    ) -> None:
        """
        Log `torch.Tensor`-like to data to the current exp dir.

        Currently supports:
            - `.npy`

        :param name: name of the image
        :param img_like: image to log
        :param subdir: subdirectory to save to
        """

        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        # optionally, save in a subdir
        if subdir is not None:
            outdir = os.path.join(outdir, subdir)
            os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # TODO: support other data formats
        if name.endswith(".npy"):
            np.save(out_fp, data)

    def log_colorized_tensors(
        self, *samples: Tuple[torch.Tensor, str], file_name: str
    ) -> plt.Figure:
        """
        Log tensors with the exact shape: [B, H, W], using an added color pallet to make things pretty.
        """

        MAX_COLS = 3
        IMAGE_SIZE_IN = 6
        num_images = len(samples)
        n_cols = min(num_images, MAX_COLS)
        n_rows = math.ceil(num_images / MAX_COLS)

        # TODO: is 4-inches enough?... (;
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * IMAGE_SIZE_IN, n_rows * IMAGE_SIZE_IN)
        )

        # axes always 2d arr
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for idx, (tensor, name) in enumerate(samples):
            row = idx // MAX_COLS
            col = idx % MAX_COLS
            # only use first tensor in batch
            img = tensor[0, ...]
            # strange, convert to img like returns a list...
            img = convert_to_img_like(img)[0]
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(name, fontsize=14)
            ax.axis("off")

        # turn off extra subplots
        # idk, chat thinks this a good idea
        total_cells = n_rows * n_cols
        for idx in range(num_images, total_cells):
            row = idx // MAX_COLS
            col = idx % MAX_COLS
            axes[row, col].axis("off")

        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, file_name)
        plt.savefig(out_fp, bbox_inches="tight", pad_inches=0.1, dpi=300)
        return fig

    def log_original_masked_predicted_sample_triplet(
        self,
        y: torch.Tensor,
        y_sparse: torch.Tensor,
        y_hat: torch.Tensor,
        name: str,
    ) -> None:
        """
        Expect inputs with shapes (B, H, W).
        """
        # (B, H, W) -> (H, W)
        y = y[0, ...]
        y_sparse = y_sparse[0, ...]
        y_hat = y_hat[0, ...]
        # (H, W) -> (H, W, C)
        y, y_sparse, y_hat = convert_to_img_like(y, y_sparse, y_hat)
        combined_image = np.concatenate([y, y_sparse, y_hat], axis=1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_image)
        ax.axis("off")
        h, w = y.shape[:2]
        labels = ["Original", "Masked", "Predicted"]
        for i, label in enumerate(labels):
            x_pos = i * w + w // 2
            ax.text(x_pos, -4, label, fontsize=14, ha="center", color="black")
        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        plt.savefig(out_fp, bbox_inches="tight", pad_inches=0.1, dpi=300)

    def log_original_masked_predicted_sample_triplet_controlnet(
        self,
        y: torch.Tensor,
        y_sparse: torch.Tensor,
        y_hat: torch.Tensor,
        name: str,
    ) -> None:
        """
        Expect img-like inputs with shapes (H, W, C).
        """
        combined_image = np.concatenate([y, y_sparse, y_hat], axis=1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_image)
        ax.axis("off")
        h, w = y.shape[:2]
        labels = ["Original", "Masked", "Predicted"]
        for i, label in enumerate(labels):
            x_pos = i * w + w // 2
            ax.text(x_pos, -4, label, fontsize=14, ha="center", color="black")
        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        plt.savefig(out_fp, bbox_inches="tight", pad_inches=0.1, dpi=300)
