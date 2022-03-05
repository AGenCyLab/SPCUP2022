import sys
import pathlib
from typing import Tuple

ROOT = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT)

import torch
import numpy as np
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset


class SPCUP22Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        annotations_file_name: str = "labels.csv",
        audio_duration: int = 6,
        mode: str = "training",
    ):

        if mode not in ("training", "eval"):
            raise Exception(
                "Unknown mode '{}', expected one of 'training' or 'eval'".format(
                    mode
                )
            )

        self.mode = mode
        self.dataset_root = pathlib.Path(dataset_root)

        if self.mode == "eval":
            annotations_file_name = "labels_eval_part1.csv"

        self.annotations_csv = str(
            self.dataset_root.joinpath(annotations_file_name)
        )
        self.annotations_df = pd.read_csv(self.annotations_csv)
        self.num_samples = self.annotations_df["track"].shape[0]
        self.duration = audio_duration

    def __len__(self):
        return self.num_samples

    def read_audio_file(self, audio_path: str) -> np.ndarray:
        """
        Reads a given audio file. Slices it up or trims it down if the length 
        is not exactly equal to self.duration. By default, max duration is 6 
        seconds according to the paper
        """
        audio, sample_rate = sf.read(audio_path)

        # padding
        if len(audio) < self.duration * sample_rate:
            audio = np.tile(
                audio, int((self.duration * sample_rate) // len(audio)) + 1
            )

        # trim
        audio = audio[0 : (int(self.duration * sample_rate))]

        audio = np.expand_dims(audio, axis=0)

        return np.asarray(audio, dtype=np.float32)

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        if self.mode == "training":
            label = self.annotations_df.iloc[index, 1]
            filename = self.annotations_df.iloc[index, 0]
            filepath = str(self.dataset_root.joinpath(filename))
            audio = self.read_audio_file(filepath)

            return audio, label
        elif self.mode == "eval":
            # evaluation csv has no labels and the filenames are at the
            # 1-th index
            filename = self.annotations_df.iloc[index, 1]
            filepath = str(self.dataset_root.joinpath(filename))
            audio = self.read_audio_file(filepath)
            return audio, None

