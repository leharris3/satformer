import os
import torch
import h5py
import random
import numpy as np
import xarray as xr

from typing import List
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from src.util.scale import scale_zero_to_one


WFC_ROOT_DIR = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data"

# HACK: hard code normalization stats from training set
X_MAX     = 336.2159
Y_MAX     = 603.4000
Y_REG_MAX = 536120.9375


class Sat2RadDataset(Dataset):

    def __init__(
            self, 
            split="train",
            steps_per_epoch:int=100,
            toy_dataset=False,
            X_max=X_MAX,
            y_max=Y_MAX,
            y_reg_max=Y_REG_MAX,
        ):

        super().__init__()

        self.split = split
        self.steps_per_epoch = steps_per_epoch

        if not split in ['train', 'val', 'test']: 
            raise Exception(f"Invalid split: {split}")
        
        opera_regex   = f"{WFC_ROOT_DIR}/*/*/OPERA/*{split}*.h5"
        hrit_regex    = f"{WFC_ROOT_DIR}/*/*/HRIT/*{split}*ns.h5"
        hrit_regex_ii = f"{WFC_ROOT_DIR}/*/*/*/HRIT/*{split}*ns.h5"
        
        all_opera_fps = list(set(glob(opera_regex)))

        # please don't judge mighty lord in heaven (:
        all_hrit_fps  = list(set(list(set(glob(hrit_regex))) + list(set(glob(hrit_regex_ii)))))

        all_fp_dict = {}
        fp_dict     = {}

        # please don't looooookkk...!!!
        # remove some symlinks
        for op_fp, hr_fp in zip(all_opera_fps, all_hrit_fps):
            
            if not Path(op_fp).is_symlink():

                name = Path(op_fp).name
                name = name.split(".")[0]
                if name not in all_fp_dict:
                    all_fp_dict[name] = {
                        "opera": op_fp,
                        "hrit" : None,
                    }
                else:
                    all_fp_dict[name]['opera'] = op_fp

            if not Path(hr_fp).is_symlink():

                name = Path(hr_fp).name
                name = name.split(".")[0]

                if name not in all_fp_dict:
                    all_fp_dict[name] = {
                        "opera": None,
                        "hrit" : hr_fp,
                    }
                else:
                    all_fp_dict[name]['hrit'] = hr_fp

        hrit_fps  = []
        opera_fps = []

        for k, v in all_fp_dict.items():
            if v['opera'] != None and v['hrit'] != None: 
                fp_dict[k] = v
                hrit_fps.append(v['hrit'])
                opera_fps.append(v['opera'])

        # NOTE: not implemented
        if toy_dataset:
            opera_fps = opera_fps[:1]
            hrit_fps  = hrit_fps[:1]
        
        # multiband satallite data
        # [B, T1, 11, 252, 252]
        self.hrit_buffer: List[xr.Dataset] = []

        # high-res radar rain-rates
        # [B, T2, 252, 252, 1]
        self.opera_buffer: List[xr.Dataset] = []

        for fp_hr, fp_op in tqdm(zip(hrit_fps, opera_fps), desc="Loading dataset...", total=len(hrit_fps)):
            
            ds = xr.open_dataset(fp_hr, phony_dims='sort')
            self.hrit_buffer.append(ds)

            ds = xr.open_dataset(fp_op, phony_dims='sort')
            self.opera_buffer.append(ds)

        # TODO: quickly calculate rough std normal statistics for X and y sets
        self.X_max = X_max
        self.y_max = y_max
        self.y_reg_max = y_reg_max

    def __len__(self) -> int: 
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> dict:
        
        # TODO: randomly crop a H=32, W=32 slice from X, y

        # choose rand idx in [0, ..., # datasets)
        rand_idx = random.randint(0, len(self.hrit_buffer) - 1)

        # [T1, 11, 252, 252]
        hrit_ds  = self.hrit_buffer[rand_idx]['REFL-BT']
        
        # [T2, 1, 252, 252]; T2<=T1 typically
        opera_ds = self.opera_buffer[rand_idx]['rates.crop']

        T       = min(hrit_ds.shape[0], opera_ds.shape[0])
        start_T = random.randint(0, T-20)

        # input: 1H satellite data
        X = hrit_ds[start_T:start_T+4, ...].to_numpy()
        X = torch.Tensor(X)
        
        # HACK: [T=4, C=11, H, W] -> [C=11, H, W]
        X = X.mean(dim=0)

        # label: 4H proceeding rainfall
        # [T=16, C=1, H=252, W=252]
        y = opera_ds[start_T+4:start_T+20].to_numpy()
        y = torch.Tensor(y)

        # NOTE: clip @0; it can't rain a negative amount; large negative values in our datasets
        y = y.clip(0)

        # wfc challenge #1 target
        # - average hourly cummulative rainfall
        # - individual feature maps (H, W) are 15 minute accumulated rainfall
        # - we derive regression targets: 
        # ---- hourly rainfall    : H = (maps) * 4
        # ---- avg hourly rainfall: H/16
        
        # [T, C=1, H, W] -> [T, H, W]
        y_reg = y.squeeze(1)
        y_reg = y_reg.sum(dim=(1, 2)) # [T]; cummulative 15M rainfall
        y_reg = y_reg * 4             # [T]; cummulative 1H rainfall
        y_reg = y_reg.mean()          # [T]; average cummulative 1H rainfall

        X_norm     = scale_zero_to_one(X,     dataset_min=0, dataset_max=self.X_max)
        y_norm     = scale_zero_to_one(y,     dataset_min=0, dataset_max=self.y_max)
        y_reg_norm = scale_zero_to_one(y_reg, dataset_min=0, dataset_max=self.y_reg_max)

        # input (X)
        # 1H HRIT satallite context (can be larger); centered about corresponding area of precipitation
        # - (B, H, W, C, T) -> (B, (32 // 6) + (32 // 6) + 1, 11, 4) -> (B, 6+, 6+, 11, 4)

        # output (y)
        # OPERA average, hourly cummulative precipitation for 4H; 32x32 pixels
        # - (B, H, W, C, T) -> (B, 32, 32, 1, 16); layer_last
        # - regression target: (layer_last).mean() * 4; note: we do NOT average across batch dimension

        return {
            "X"     : X,
            "y"     : y,
            "y_reg" : y_reg,
            "X_norm": X_norm,
            "y_norm": y_norm,
            "y_reg_norm": y_reg_norm
        }


class ChallengeOneTestSet(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(): return -1

    def __getitem__(self, index):
        return {}
    

if __name__ == "__main__":

    ds   = Sat2RadDataset(split="val", toy_dataset=False)
    item = ds.__getitem__(0)
    breakpoint()