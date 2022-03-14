import sys
import pathlib
from typing import Tuple

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.audio import read_audio_file
from torchvision.transforms import Compose


class SPCUP22Dataset(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        max_duration: float = 6.0,
        mode: str = "training",
        transform: Compose = None,
    ):

        if mode not in ("training", "eval"):
            raise Exception(
                "Unknown mode '{}', expected one of 'training' or 'eval'".format(
                    mode
                )
            )

        self.mode = mode
        self.transform = transform

        self.annotations_df = annotations_df
        self.num_samples = self.annotations_df["track"].shape[0]
        self.max_duration = max_duration

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        label = None

        if self.mode == "training":
            label = self.annotations_df.iloc[index, 1]
            filepath = self.annotations_df.iloc[index, 0]
        elif self.mode == "eval":
            # evaluation csv has no labels and the filenames are at the
            # 1-th index
            filepath = self.annotations_df.iloc[index, 1]

        audio = read_audio_file(filepath, duration=self.max_duration)

        if self.transform:
            audio, _ = self.transform((audio, label))

        return audio, label

