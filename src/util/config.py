import os
import yaml
import torch
import torch.nn as nn

from typing import List, Optional
from src.util.loss import DiceLoss, FocalLoss, VAELoss, ImageInpaintingL1Loss
from torchvision.models import resnet152, swin_b, efficientnet_v2_l, vit_l_16
from torchvision.models import (
    ResNet152_Weights,
    Swin_B_Weights,
    EfficientNet_V2_L_Weights,
    ViT_L_16_Weights,
)
from src.models.simple_z_predictor import SimpleZRegressionVisionTransformer
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.unet.unet import UNet, ThickUNet
from src.models.unetr.unetr import UNETR
from src.models.classic_recon import (
    LinearInterpolationInpainter,
    BicubicInterpolationInpainter,
    AMPInpainter,
    NearestNeighborsInpainter,
)
from src.models.our_method.swin_cafm import SwinCAFM
from src.models.prev_methods.sstem import SSTEM
from src.models.prev_methods.gpstruct import GPSTRUCT


def parse_config(fp: str) -> dict:
    r"""
    Args
        :param fp: path to config file
    Returns
        :return: dict
    """
    assert os.path.isfile(fp), f"Error: config file @ {fp} does not exist"
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


LOSS_FUNCTIONS = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "Dice": DiceLoss,
    "Focal": FocalLoss,
    "Huber": nn.HuberLoss,
    "VAE": VAELoss,
    "InpaintingL1": ImageInpaintingL1Loss,
}

MODELS = {
    "ours": {
        "fn": SwinCAFM.get,
        "weights": None,
    },
    "simple_z_reg_vit": {
        "fn": SimpleZRegressionVisionTransformer.get,
        "weights": None,
    },
    "resnet152": {
        "fn": resnet152,
        "weights": ResNet152_Weights.IMAGENET1K_V2,
    },
    "swin_b": {
        "fn": swin_b,
        "weights": Swin_B_Weights.IMAGENET1K_V1,
    },
    "efficientnet_v2_l": {
        "fn": efficientnet_v2_l,
        "weights": EfficientNet_V2_L_Weights.IMAGENET1K_V1,
    },
    "vit_l_16": {
        "fn": vit_l_16,
        "weights": ViT_L_16_Weights.IMAGENET1K_V1,
    },
    "ae": {"fn": Autoencoder.get, "weights": None},
    "vae": {"fn": VAE.get, "weights": None},
    "unet": {"fn": UNet.get, "weights": None},
    "thick_unet": {"fn": ThickUNet.get, "weights": None},
    "unetr": {"fn": UNETR.get, "weights": None},
    "linear_interpolation": {"fn": LinearInterpolationInpainter.get, "weights": None},
    "bicubic_interpolation": {"fn": BicubicInterpolationInpainter.get, "weights": None},
    "amp_interpolation": {"fn": AMPInpainter.get, "weights": None},
    "nn_interpolation": {"fn": NearestNeighborsInpainter.get, "weights": None},
    "sstem_interpolation": {"fn": SSTEM.get, "weights": None},
    "gpstruct_interpolation": {"fn": GPSTRUCT.get, "weights": None},
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}


class TrainConfig:
    """
    Object representing a config file for a training run of a model.
    """

    def __init__(self, config_fp: str):

        if not os.path.isfile(config_fp):
            raise FileNotFoundError(f"Config file not found: {config_fp}")

        config_dict: dict = parse_config(config_fp)

        # --- Global settings ---
        global_cfg: dict = config_dict.get("global", {})
        self.device: int = global_cfg.get("device", 0)
        self.mode: str = global_cfg.get("mode", "train")
        self.formulation: Optional[str] = global_cfg.get("formulation", None)

        # --- Model settings ---
        model_cfg: dict = config_dict.get("model", {})
        self.model_name: str = model_cfg.get("name", "")
        self.pretrained: str = model_cfg.get("pretrained", False)
        self.weights: str = model_cfg.get("weights", None)
        self.model_config_file: str = model_cfg.get("config", None)

        # --- Surrogate settings ---
        surrogate_cfg: dict = config_dict.get("surrogate", {})
        self.surrogate_name: str = surrogate_cfg.get("name", "")
        self.surgate_weights: str = surrogate_cfg.get("weights", None)

        # --- Training settings ---
        training_cfg: str = config_dict.get("training", {})
        self.train_batch_size: int = training_cfg.get("batch_size", 1)
        self.steps_per_epoch: int = training_cfg.get("steps_per_epoch", 1024)
        self.epochs: int = training_cfg.get("epochs", 2000)
        self.train_loss: Optional[str] = training_cfg.get("loss", None)
        self.learning_rate: str = training_cfg.get("lr", 1e-4)
        self.optimizer: str = training_cfg.get("optimizer", "Adam")

        # --- Validation settings ---
        validation_cfg: dict = config_dict.get("validation", {})
        self.val_batch_size: int = validation_cfg.get("batch_size", 1)
        self.val_steps_per_epoch: int = validation_cfg.get("steps_per_epoch", 256)
        self.val_loss: Optional[str] = validation_cfg.get("loss", None)

        # --- Dataset settings ---
        dataset_cfg: dict = config_dict.get("dataset", {})
        self.dataset_name: str = dataset_cfg.get("name", "")
        self.image_size: Optional[str] = dataset_cfg.get("image_size", None)
        self.crop_size: Optional[str] = dataset_cfg.get("crop_size", None)
        self.num_workers: int = dataset_cfg.get("num_workers", 0)
        self.masking_ratio: int = dataset_cfg.get("masking_ratio", 1)

        # --- Logging settings ---
        logging_cfg: dict = config_dict.get("logging", {})
        self.log_root: str = logging_cfg.get("root", "")
        self.exp_name: str = logging_cfg.get("exp_name", "")
        self.result_columns: List = logging_cfg.get("result_columns", [])
        self.save_weights: bool = logging_cfg.get("save_weights", True)
        self.save_only_best_weights: bool = logging_cfg.get(
            "save_only_best_weights", True
        )
        self.enable_tensorboard: bool = logging_cfg.get("enable_tensorboard", False)
        self.log_figures: bool = logging_cfg.get("log_figures", False)
        self.log_interval: int = logging_cfg.get("log_interval", 1)

    def to_dict(self) -> dict:
        config_dict = {
            "global": {
                "device": self.device,
                "mode": self.mode,
                "formulation": self.formulation,
            },
            "model": {
                "name": self.model_name,
                "pretrained": self.pretrained,
                "weights": self.weights,
                "config": self.model_config_file,
            },
            "surrogate": {
                "name": self.surrogate_name,
                "weights": self.surgate_weights,
            },
            "training": {
                "batch_size": self.train_batch_size,
                "steps_per_epoch": self.steps_per_epoch,
                "epochs": self.epochs,
                "loss": self.train_loss,
                "lr": self.learning_rate,
                "optimizer": self.optimizer,
            },
            "validation": {
                "batch_size": self.val_batch_size,
                "steps_per_epoch": self.val_steps_per_epoch,
                "loss": self.val_loss,
            },
            "dataset": {
                "name": self.dataset_name,
                "image_size": self.image_size,
                "crop_size": self.crop_size,
                "num_workers": self.num_workers,
                "masking_ratio": self.masking_ratio,
            },
            "logging": {
                "root": self.log_root,
                "exp_name": self.exp_name,
                "result_columns": self.result_columns,
                "save_weights": self.save_weights,
                "save_only_best_weights": self.save_only_best_weights,
                "enable_tensorboard": self.enable_tensorboard,
                "log_figures": self.log_figures,
                "log_interval": self.log_interval,
            },
        }
        return config_dict

    def save_config(self, output_fp: str) -> None:
        """
        Save the current configuration to a YAML file, preserving the original format.
        """
        config_dict = {
            "global": {
                "device": self.device,
                "mode": self.mode,
                "formulation": self.formulation,
            },
            "model": {
                "name": self.model_name,
                "pretrained": self.pretrained,
                "weights": self.weights,
                "config": self.model_config_file,
            },
            "surrogate": {
                "name": self.surrogate_name,
                "weights": self.surgate_weights,
            },
            "training": {
                "batch_size": self.train_batch_size,
                "steps_per_epoch": self.steps_per_epoch,
                "epochs": self.epochs,
                "loss": self.train_loss,
                "lr": self.learning_rate,
                "optimizer": self.optimizer,
            },
            "validation": {
                "batch_size": self.val_batch_size,
                "steps_per_epoch": self.val_steps_per_epoch,
                "loss": self.val_loss,
            },
            "dataset": {
                "name": self.dataset_name,
                "image_size": self.image_size,
                "crop_size": self.crop_size,
                "num_workers": self.num_workers,
                "masking_ratio": self.masking_ratio,
            },
            "logging": {
                "root": self.log_root,
                "exp_name": self.exp_name,
                "result_columns": self.result_columns,
                "save_weights": self.save_weights,
                "save_only_best_weights": self.save_only_best_weights,
                "enable_tensorboard": self.enable_tensorboard,
                "log_figures": self.log_figures,
                "log_interval": self.log_interval,
            },
        }
        # save config
        with open(output_fp, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


class EvalConfig:
    """
    Object representing a config file for a evaluation run of a model.
    """

    def __init__(self, config_fp: str):

        if not os.path.isfile(config_fp):
            raise FileNotFoundError(f"Config file not found: {config_fp}")

        config_dict: dict = parse_config(config_fp)

        # --- Global settings ---
        global_cfg: dict = config_dict.get("global", {})
        self.device: int = global_cfg.get("device", 0)
        self.mode: str = global_cfg.get("mode", "train")
        self.formulation: Optional[str] = global_cfg.get("formulation", None)

        # --- Model settings ---
        model_cfg: dict = config_dict.get("model", {})
        self.model_name: str = model_cfg.get("name", "")
        self.pretrained: str = model_cfg.get("pretrained", False)
        self.weights: str = model_cfg.get("weights", None)
        self.model_config_file: str = model_cfg.get("config", None)

        # --- Validation settings ---
        validation_cfg: dict = config_dict.get("validation", {})
        self.val_batch_size: int = validation_cfg.get("batch_size", 1)
        self.val_steps_per_epoch: int = validation_cfg.get("steps_per_epoch", 256)
        self.val_loss: Optional[str] = validation_cfg.get("loss", None)

        # --- Dataset settings ---
        dataset_cfg: dict = config_dict.get("dataset", {})
        self.dataset_name: str = dataset_cfg.get("name", "")
        self.image_size: Optional[str] = dataset_cfg.get("image_size", None)
        self.crop_size: Optional[str] = dataset_cfg.get("crop_size", None)
        self.num_workers: int = dataset_cfg.get("num_workers", 0)
        self.masking_ratio: int = dataset_cfg.get("masking_ratio", 1)

        # --- Logging settings ---
        logging_cfg: dict = config_dict.get("logging", {})
        self.log_root: str = logging_cfg.get("root", "")
        self.exp_name: str = logging_cfg.get("exp_name", "")
        self.result_columns: List = logging_cfg.get("result_columns", [])
        self.save_weights: bool = logging_cfg.get("save_weights", True)
        self.save_only_best_weights: bool = logging_cfg.get(
            "save_only_best_weights", True
        )
        self.enable_tensorboard: bool = logging_cfg.get("enable_tensorboard", False)
        self.log_figures: bool = logging_cfg.get("log_figures", False)
        self.log_interval: int = logging_cfg.get("log_interval", 1)

    def to_dict(self) -> dict:
        config_dict = {
            "global": {
                "device": self.device,
                "mode": self.mode,
                "formulation": self.formulation,
            },
            "model": {
                "name": self.model_name,
                "pretrained": self.pretrained,
                "weights": self.weights,
                "config": self.model_config_file,
            },
            "validation": {
                "batch_size": self.val_batch_size,
                "steps_per_epoch": self.val_steps_per_epoch,
                "loss": self.val_loss,
            },
            "dataset": {
                "name": self.dataset_name,
                "image_size": self.image_size,
                "crop_size": self.crop_size,
                "num_workers": self.num_workers,
                "masking_ratio": self.masking_ratio,
            },
            "logging": {
                "root": self.log_root,
                "exp_name": self.exp_name,
                "result_columns": self.result_columns,
                "save_weights": self.save_weights,
                "save_only_best_weights": self.save_only_best_weights,
                "enable_tensorboard": self.enable_tensorboard,
                "log_figures": self.log_figures,
                "log_interval": self.log_interval,
            },
        }
        return config_dict

    def save_config(self, output_fp: str) -> None:
        """
        Save the current configuration to a YAML file, preserving the original format.
        """
        config_dict = {
            "global": {
                "device": self.device,
                "mode": self.mode,
                "formulation": self.formulation,
            },
            "model": {
                "name": self.model_name,
                "pretrained": self.pretrained,
                "weights": self.weights,
                "config": self.model_config_file,
            },
            "validation": {
                "batch_size": self.val_batch_size,
                "steps_per_epoch": self.val_steps_per_epoch,
                "loss": self.val_loss,
            },
            "dataset": {
                "name": self.dataset_name,
                "image_size": self.image_size,
                "crop_size": self.crop_size,
                "num_workers": self.num_workers,
                "masking_ratio": self.masking_ratio,
            },
            "logging": {
                "root": self.log_root,
                "exp_name": self.exp_name,
                "result_columns": self.result_columns,
                "save_weights": self.save_weights,
                "save_only_best_weights": self.save_only_best_weights,
                "enable_tensorboard": self.enable_tensorboard,
                "log_figures": self.log_figures,
                "log_interval": self.log_interval,
            },
        }
        # save config
        with open(output_fp, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


class SurrogateEvalConfig:
    """
    Object representing a config file for an evaluation run of an older-surrogate model pair.
    """

    def __init__(self, config_fp: str):

        if not os.path.isfile(config_fp):
            raise FileNotFoundError(f"Config file not found: {config_fp}")

        config_dict: dict = parse_config(config_fp)

        # --- Global settings ---
        global_cfg: dict = config_dict.get("global", {})
        self.device: int = global_cfg.get("device", 0)
        self.mode: str = global_cfg.get("mode", "train")
        self.formulation: Optional[str] = global_cfg.get("formulation", None)

        # --- Denoising model settings ---
        denoising_model_cfg: dict = config_dict.get("denoising_model", {})
        self.denoising_model_name: str = denoising_model_cfg.get("name", "")
        self.denoising_model_pretrained: bool = denoising_model_cfg.get(
            "pretrained", False
        )
        self.denoising_model_weights: Optional[str] = denoising_model_cfg.get(
            "weights", None
        )
        self.denoising_model_config_file: Optional[str] = denoising_model_cfg.get(
            "config", None
        )

        # --- Older surrogate model settings ---
        older_surrogate_model_cfg: dict = config_dict.get("older_surrogate_model", {})
        self.older_surrogate_model_name: str = older_surrogate_model_cfg.get("name", "")
        self.older_surrogate_model_pretrained: bool = older_surrogate_model_cfg.get(
            "pretrained", False
        )
        self.older_surrogate_model_weights: Optional[str] = (
            older_surrogate_model_cfg.get("weights", None)
        )
        self.older_surrogate_model_config_file: Optional[str] = (
            older_surrogate_model_cfg.get("config", None)
        )

        # --- Training settings ---
        training_cfg: dict = config_dict.get("training", {})
        self.train_batch_size: int = training_cfg.get("batch_size", 1)
        self.train_steps_per_epoch: int = training_cfg.get("steps_per_epoch", 1024)
        self.epochs: int = training_cfg.get("epochs", 100)
        self.train_loss: Optional[str] = training_cfg.get("loss", None)
        self.lr: float = training_cfg.get("lr", 1e-4)
        self.optimizer: Optional[str] = training_cfg.get("optimizer", "Adam")

        # --- Validation settings ---
        validation_cfg: dict = config_dict.get("validation", {})
        self.val_batch_size: int = validation_cfg.get("batch_size", 1)
        self.val_steps_per_epoch: int = validation_cfg.get("steps_per_epoch", 256)
        self.val_loss: Optional[str] = validation_cfg.get("loss", None)

        # --- Dataset settings ---
        dataset_cfg: dict = config_dict.get("dataset", {})
        self.dataset_name: str = dataset_cfg.get("name", "")
        self.image_size: Optional[int] = dataset_cfg.get("image_size", None)
        self.crop_size: Optional[int] = dataset_cfg.get("crop_size", None)
        self.num_workers: int = dataset_cfg.get("num_workers", 0)
        self.masking_ratio: int = dataset_cfg.get("masking_ratio", 1)

        # --- Logging settings ---
        logging_cfg: dict = config_dict.get("logging", {})
        self.log_root: str = logging_cfg.get("root", "")
        self.exp_name: str = logging_cfg.get("exp_name", "")
        self.result_columns: List = logging_cfg.get("result_columns", [])
        self.save_weights: bool = logging_cfg.get("save_weights", True)
        self.save_only_best_weights: bool = logging_cfg.get(
            "save_only_best_weights", True
        )
        self.enable_tensorboard: bool = logging_cfg.get("enable_tensorboard", False)
        self.log_figures: bool = logging_cfg.get("log_figures", False)
        self.log_interval: int = logging_cfg.get("log_interval", 1)

    def to_dict(self) -> dict:
        """
        Return the configuration as a dictionary matching the YAML file structure.
        """
        config_dict = {
            "global": {
                "device": self.device,
                "mode": self.mode,
                "formulation": self.formulation,
            },
            "denoising_model": {
                "name": self.denoising_model_name,
                "pretrained": self.denoising_model_pretrained,
                "weights": self.denoising_model_weights,
                "config": self.denoising_model_config_file,
            },
            "older_surrogate_model": {
                "name": self.older_surrogate_model_name,
                "pretrained": self.older_surrogate_model_pretrained,
                "weights": self.older_surrogate_model_weights,
                "config": self.older_surrogate_model_config_file,
            },
            "training": {
                "batch_size": self.train_batch_size,
                "steps_per_epoch": self.train_steps_per_epoch,
                "epochs": self.epochs,
                "loss": self.train_loss,
                "lr": self.lr,
                "optimizer": self.optimizer,
            },
            "validation": {
                "batch_size": self.val_batch_size,
                "steps_per_epoch": self.val_steps_per_epoch,
                "loss": self.val_loss,
            },
            "dataset": {
                "name": self.dataset_name,
                "image_size": self.image_size,
                "crop_size": self.crop_size,
                "num_workers": self.num_workers,
                "masking_ratio": self.masking_ratio,
            },
            "logging": {
                "root": self.log_root,
                "exp_name": self.exp_name,
                "result_columns": self.result_columns,
                "save_weights": self.save_weights,
                "save_only_best_weights": self.save_only_best_weights,
                "enable_tensorboard": self.enable_tensorboard,
                "log_figures": self.log_figures,
                "log_interval": self.log_interval,
            },
        }
        return config_dict

    def save_config(self, output_fp: str) -> None:
        """
        Save the current configuration to a YAML file, preserving the original structure.
        """
        config_dict = self.to_dict()
        with open(output_fp, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


class ModelConfig:
    """
    Object representing a config file for a SwinIR model.
    """

    def __init__(self, config_fp: str):

        if not os.path.isfile(config_fp):
            raise FileNotFoundError(f"Config file not found: {config_fp}")

        config_dict: dict = parse_config(config_fp)

        # --- Top-level setting: weights_fp ---
        self.weights_fp: str = config_dict.get("weights_fp", "")

        # --- Hyperparameters ---
        hyperparams: dict = config_dict.get("hyperparams", {})
        self.upscale: int = hyperparams.get("upscale", 8)
        self.img_size: List[int] = hyperparams.get("img_size", [128, 128])
        self.window_size: int = hyperparams.get("window_size", 8)
        self.img_range: float = hyperparams.get("img_range", 1.0)
        self.depths: List[int] = hyperparams.get("depths", [8, 8, 8, 8, 8, 8])
        self.embed_dim: int = hyperparams.get("embed_dim", 180)
        self.num_heads: List[int] = hyperparams.get("num_heads", [6, 6, 6, 6, 6, 6])
        self.mlp_ratio: int = hyperparams.get("mlp_ratio", 2)
        self.upsampler: str = hyperparams.get("upsampler", "no_upscale")
        self.resi_connection: str = hyperparams.get("resi_connection", "1conv")
        self.drop_path_rate = (
            config_dict.get("hyperparams", {}).get("drop_path_rate", 0.1),
        )

        # layer norm
        self.layer_norm_str = config_dict.get("hyperparams", {}).get("norm_layer", None)
        layer_norm = (
            torch.nn.LayerNorm if self.layer_norm_str == "torch.nn.LayerNorm" else None
        )
        self.norm_layer = layer_norm

    def to_dict(self) -> dict:
        config_dict = {
            "weights_fp": self.weights_fp,
            "hyperparams": {
                "upscale": self.upscale,
                "img_size": self.img_size,
                "window_size": self.window_size,
                "img_range": self.img_range,
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "drop_path_rate": self.drop_path_rate,
                "norm_layer": self.layer_norm_str,
                "upsampler": self.upsampler,
                "resi_connection": self.resi_connection,
            },
        }
        return config_dict

    def save_config(self, output_fp: str) -> None:
        """
        Save the current configuration to a YAML file, preserving the original format.
        """
        config_dict = {
            "weights_fp": self.weights_fp,
            "hyperparams": {
                "upscale": self.upscale,
                "img_size": self.img_size,
                "window_size": self.window_size,
                "img_range": self.img_range,
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "upsampler": self.upsampler,
                "resi_connection": self.resi_connection,
            },
        }
        with open(output_fp, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
