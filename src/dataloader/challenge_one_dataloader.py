import os
import torch
import h5py

from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


# challenge num.1 is to predict cummulative rainfall from satallite data
WFC_ROOT_DIR = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data"

# OPERA_TRAIN_DATA_PATH = "weather4cast_data/w4c25/*/OPERA/*.train.*.h5"
# OPERA_VAL_DATA_PATH   = "weather4cast_data/w4c25/*/OPERA/*.val.*.h5"
# HRIT_TRAIN_DATA_PATH  = "weather4cast_data/w4c25/*/HRIT/*.train.*.h5"
# HRIT_VAL_DATA_PATH    = "weather4cast_data/w4c25/*/HRIT/*.val.*.h5"


class Sat2RadDataset(Dataset):

    def __init__(self, split="train", toy_dataset=False):

        super().__init__()

        self.split = split

        if not split in ['train', 'val', 'test']: 
            raise Exception(f"Invalid split: {split}")
        
        opera_regex   = f"{WFC_ROOT_DIR}/*/*/OPERA/*{split}*.h5"
        hrit_regex    = f"{WFC_ROOT_DIR}/*/*/HRIT/*{split}*ns.h5"
        hrit_regex_ii = f"{WFC_ROOT_DIR}/*/*/*/HRIT/*{split}*ns.h5"
        
        all_opera_fps = list(set(glob(opera_regex))) 
        all_hrit_fps  = list(set(list(set(glob(hrit_regex))) + list(set(glob(hrit_regex_ii)))))

        opera_fps = []
        hrit_fps  = []

        # remove some symlinks
        for op_fp, hr_fp in zip(all_opera_fps, all_hrit_fps):
            if not Path(op_fp).is_symlink(): opera_fps.append(op_fp)
            if not Path(hr_fp).is_symlink(): hrit_fps.append(hr_fp)

        breakpoint()

        if toy_dataset:
            opera_fps = opera_fps[:1]
            hrit_fps  = hrit_fps[:1]
        
        # high-res radar rain-rates
        # [B, 226, 226, 1, T]
        self.opera_buffer = []

        # multiband satallite data
        # [B, 1512, 1512, 11, T]
        self.hrit_buffer = []

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