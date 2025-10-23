import os
import torch
import h5py

from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


# challenge num.1 is to predict cummulative rainfall from satallite data
OPERA_TRAIN_DATA_PATH = "weather4cast_data/w4c25/*/OPERA/*.train.*.h5"
OPERA_VAL_DATA_PATH   = "weather4cast_data/w4c25/*/OPERA/*.val.*.h5"
HRIT_TRAIN_DATA_PATH  = "weather4cast_data/w4c25/*/HRIT/*.train.*.h5"
HRIT_VAL_DATA_PATH    = "weather4cast_data/w4c25/*/HRIT/*.val.*.h5"


class Sat2RadDataset(Dataset):

    def __init__(self, split="train", toy_dataset=False):

        super().__init__()

        opera_regex = ""
        hrit_regex  = ""

        if split == "train":
            opera_regex = OPERA_TRAIN_DATA_PATH
            hrit_regex  = HRIT_VAL_DATA_PATH
        elif split == "val":
            opera_regex = OPERA_VAL_DATA_PATH
            hrit_regex  = HRIT_VAL_DATA_PATH
        else:
            raise Exception(f"Invalid split: {split}")
        
        opera_fps = glob(opera_regex)
        hrit_fps  = glob(hrit_regex)

        if toy_dataset:
            opera_fps = opera_fps[:1]
            hrit_fps  = hrit_fps[:1]
        
        # high-res radar rain-rates
        # [B, 226, 226, 1, T]
        self.opera_buffer = []

        # multiband satallite data
        # [B, 1512, 1512, 11, T]
        self.hrit_buffer = []

        # for fp in tqdm(opera_fps, total=len(opera_fps), desc=f"Loading OPERA data..."):
            
        #     with h5py.File(fp, "r") as f:
        #         dataset_name = list(f.keys())[0]
        #         t = f[dataset_name][:]
        #         self.opera_buffer.append(t)

        # for fp in tqdm(hrit_fps, total=len(hrit_fps), desc=f"Loading HRIT data..."):
            
        #     with h5py.File(fp, "r") as f:
        #         dataset_name = list(f.keys())[0]
        #         t = f[dataset_name][:]
        #         self.hrit_buffer.append(t)

        # safe HDF5 read
        def _read_first_dataset(fp: str):
            with h5py.File(fp, "r") as f:
                keys = sorted(list(f.keys()))
                if not keys:
                    raise RuntimeError(f"No datasets in file: {fp}")
                name = "data" if "data" in f else keys[0]
                return f[name][:]

        with ThreadPoolExecutor() as ex:

            # OPERA
            for arr in tqdm(
                ex.map(_read_first_dataset, opera_fps),
                total=len(opera_fps),
                desc="Loading OPERA data..."
            ):
                self.opera_buffer.append(torch.Tensor(arr))

            # HRIT
            for arr in tqdm(
                ex.map(_read_first_dataset, hrit_fps),
                total=len(hrit_fps),
                desc="Loading HRIT data..."
            ):
                self.hrit_buffer.append(torch.Tensor(arr))

        breakpoint()

    def __len__(): return -1

    def __getitem__(self, index):
        return {}


class ChallengeOneTestSet(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(): return -1

    def __getitem__(self, index):
        return {}
    

if __name__ == "__main__":
    ds = Sat2RadDataset(toy_dataset=True)