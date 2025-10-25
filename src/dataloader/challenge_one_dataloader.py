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


WFC_ROOT_DIR = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data"


class Sat2RadDataset(Dataset):

    def __init__(
            self, 
            split="train",
            steps_per_epoch:int=100,
            toy_dataset=False
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

    def __len__(self) -> int: 
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> dict:

        # TODO:
        # 1. sampling proceedure
        # - pick a random OPERA patch that's not too close to a boarder
        # - pick the corresponding

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

        # label: 4H proceeding rainfall
        y = opera_ds[start_T+4:start_T+20].to_numpy()
        y = torch.Tensor(y)

        # clip @0; it can't rain a negative amount; large negative values in our datasets
        y = y.clip(0)

        # input (X)
        # 1H HRIT satallite context (can be larger); centered about corresponding area of precipitation
        # - (B, H, W, C, T) -> (B, (32 // 6) + (32 // 6) + 1, 11, 4) -> (B, 6+, 6+, 11, 4)

        # output (y)
        # OPERA average, hourly cummulative precipitation for 4H; 32x32 pixels
        # - (B, H, W, C, T) -> (B, 32, 32, 1, 16); layer_last
        # - regression target: (layer_last).mean() * 4; note: we do NOT average across batch dimension

        return {
            "X": X,
            "y": y,
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