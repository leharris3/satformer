import os
import torch
import h5py
import random
import numpy as np
import xarray as xr
import pandas as pd

from typing import List
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from src.util.scale import scale_zero_to_one


WFC_ROOT_DIR    = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data"
WFC_C1_TEST_DIR = "weather4cast_data/challenge_one"


# HACK: hard code normalization stats from training set
X_MAX     = 336.2159
Y_MAX     = 603.4000
Y_REG_MAX = 536120.9375


class Sat2RadDataset(Dataset):

    def __init__(
            self, 
            split="train",
            steps_per_epoch:int=100,
            X_max=X_MAX,
            y_max=Y_MAX,
            y_reg_max=Y_REG_MAX,
        ):

        super().__init__()

        self.split = split
        if not split in ['train', 'val', 'test']: 
            raise Exception(f"Invalid split: {split}")
        
        # modify to keyword for challenge one (cum. precip) data
        if self.split == "test":
            split = "cum1test"

        self.steps_per_epoch = steps_per_epoch
        
        opera_regex   = f"{WFC_ROOT_DIR}/**/*{split}*rates*h5"
        hrit_regex    = f"{WFC_ROOT_DIR}/**/*{split}*reflbt0*h5"

        all_opera_fps = list(set(glob(opera_regex, recursive=True)))
        all_hrit_fps  = list(set(glob(hrit_regex , recursive=True)))

        self.all_fp_dict = {}
        fp_dict          = {}

        # please don't looooookkk...!!!
        # remove some symlinks
        for op_fp in all_opera_fps:
            
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

        for hr_fp in all_hrit_fps:

            if not Path(hr_fp).is_symlink():

                name = Path(hr_fp).name
                name = name.split(".")[0]

                if name not in self.all_fp_dict:
                    self.all_fp_dict[name] = {
                        "opera": None,
                        "hrit" : hr_fp,
                    }
                else:
                    self.all_fp_dict[name]['hrit'] = hr_fp

        hrit_fps  = []
        opera_fps = []

        for k, v in self.all_fp_dict.items():
            if v['opera'] != None and v['hrit'] != None: 
                fp_dict[k] = v
                hrit_fps.append(v['hrit'])
                opera_fps.append(v['opera'])

        # multiband satallite data
        # [B, T1, 11, 252, 252]
        self.hrit_buffer   : List[xr.Dataset] = []
        
        # TODO: we should just use dicts to hold all datasets
        self.hrit_test_dict: dict             = {}

        # high-res radar rain-rates
        # [B, T2, 252, 252, 1]
        self.opera_buffer: List[xr.Dataset] = []

        if self.split in ['train', 'val']:

            for fp_hr, fp_op in tqdm(zip(hrit_fps, opera_fps), desc="Loading dataset...", total=len(hrit_fps)):
                
                ds = xr.open_dataset(fp_hr, phony_dims='sort')
                self.hrit_buffer.append(ds)

                ds = xr.open_dataset(fp_op, phony_dims='sort')
                self.opera_buffer.append(ds)

        # NOTE: we only have HRIT inputs for the test set
        if self.split == "test":
            
            for k, v in self.all_fp_dict.items():
                if v['hrit'] != None:
                    fp_dict[k] = v
                    hrit_fps.append(v['hrit'])

            for fp_hr in tqdm(hrit_fps, desc="Loading dataset...", total=len(hrit_fps)):
                ds = xr.open_dataset(fp_hr, phony_dims='sort')
                self.hrit_buffer.append(ds)
                self.hrit_test_dict[Path(fp_hr).name] = ds

        if self.split == "test":
            
            csvs_rx       = f"{WFC_C1_TEST_DIR}/*.csv"
            
            # [0008, 0009, 0010]
            test_csvs_fps = sorted(glob(csvs_rx))
            dfs           = [pd.read_csv(fp) for fp in test_csvs_fps]
            
            test_df = None
            
            for fp, df in zip(test_csvs_fps, dfs):

                if test_df is None: 
                    test_df   = df
                    names     = [Path(fp).name] * len(df)
                    df['key'] = names

                else:
                    names     = [Path(fp).name] * len(df)
                    df['key'] = names
                    test_df   = pd.concat([test_df, df], axis=0)

            if self.steps_per_epoch != len(test_df): print(f"Warning: changing steps-per-epoch from {self.steps_per_epoch} -> {len(test_df)}")
            self.steps_per_epoch     = len(test_df)
            
            self.test_df             = test_df

        # TODO: quickly calculate rough std normal statistics for X and y sets
        self.X_max = X_max
        self.y_max = y_max
        self.y_reg_max = y_reg_max

    def __len__(self) -> int: 
        return self.steps_per_epoch

    def get_item_train_val(self, index: int) -> dict:

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
    
    def get_item_test(self, index: int) -> dict:
        
        assert index < self.steps_per_epoch, f"Error: get item index {index} out of bounds"

        test_row   = self.test_df.iloc[index]
        case_id    = test_row["Case-id"]
        year       = test_row["year"]
        slot_start = test_row["slot-start"]
        slot_end   = test_row["slot-end"]
        x_tl       = test_row['x-top-left']
        x_br       = test_row['x-bottom-right']
        y_tl       = test_row['y-top-left']
        y_br       = test_row['y-bottom-right']
        tr_key     = test_row['key'].split(".")[0]

        key = ""
        for k in list(self.hrit_test_dict.keys()):
            if tr_key in k: key = k; break
        
        # self.all_fp_dict
        # [T1, 11, 252, 252]
        hrit_ds  = self.hrit_test_dict[key]['REFL-BT']

        # input: 1H satellite data
        X = hrit_ds[slot_start:slot_end, ...].to_numpy()
        X = torch.Tensor(X)

        T, C, H, W  = X.shape

        # 1 HRIT PX = 6x6 OPERA pixels
        x_tl_scaled = int((x_tl / 6).item())
        x_br_scaled = int((x_br / 6).item())
        y_tl_scaled = int((y_tl / 6).item())
        y_br_scaled = int((y_br / 6).item())

        # scale window sizes to (32, 32); center crop around radar ROI
        curr_x_range = x_br_scaled - x_tl_scaled
        curr_x_diff  = (32 - curr_x_range)
        x_tl_scaled  -= (curr_x_diff // 2)
        x_br_scaled  += (32 - (x_br_scaled - x_tl_scaled))

        curr_y_range = y_br_scaled - y_tl_scaled
        curr_y_diff  = (32 - curr_y_range)
        y_tl_scaled  -= (curr_y_diff // 2)
        y_br_scaled  += (32 - (y_br_scaled - y_tl_scaled))

        # --- some samples spill out of bounds
        if x_br_scaled >= W:
            x_tl_scaled = W - 32 - 1
            x_br_scaled = W - 1

        if y_br_scaled >= H:
            y_tl_scaled = H - 32 - 1
            y_br_scaled = H - 1

        if x_tl_scaled < 0:
            x_tl_scaled = 0
            x_br_scaled = 32

        if y_tl_scaled < 0:
            y_tl_scaled = 0
            y_br_scaled = 32

        try:
            assert x_tl_scaled >= 0 and x_br_scaled < W
            assert y_tl_scaled >= 0 and y_br_scaled < H
        except:
            breakpoint()

        assert x_br_scaled - x_tl_scaled == 32
        assert y_br_scaled - y_tl_scaled == 32

        # input (X)
        # 1H HRIT satallite context (can be larger); centered about corresponding area of precipitation
        # - (B, H, W, C, T) -> (B, (32 // 6) + (32 // 6) + 1, 11, 4) -> (B, 6+, 6+, 11, 4)

        # HACK: [T=4, C=11, H, W] -> [C=11, H, W]
        X = X.mean(dim=0)

        # -> [C=11, H=32, W=32]
        X      = X[:, x_tl_scaled:x_br_scaled, y_tl_scaled:y_br_scaled]
        X_norm = scale_zero_to_one(X, dataset_min=0, dataset_max=self.X_max)

        return {
            "X"             : X,
            "X_norm"        : X_norm,
            "Case-id"       : case_id,
            "year"          : year,
            "slot-start"    : slot_start,
            "slot-end"      : slot_end,
            "x-top-left"    : x_tl,
            "x-bottom-right": x_br,
            "y-top-left"    : y_tl,
            "y-bottom-right": y_br,
        }

    def __getitem__(self, index: int) -> dict:

        if self.split in ["train", "val"]:
            return self.get_item_train_val(index)
        elif self.split == "test":
            return self.get_item_test(index)


class ChallengeOneTestSet(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(): return -1

    def __getitem__(self, index):
        return {}
    

if __name__ == "__main__":

    ds   = Sat2RadDataset(split="test")
    item = ds.__getitem__(0)
    breakpoint()