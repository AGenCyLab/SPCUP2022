from collections import defaultdict
import pathlib
import sys
from typing import Optional
import numpy as np
import pandas as pd

ROOT = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
from utils.config.load_config import load_config_file
from data_loaders.datasets.SPCUP22Dataset import SPCUP22Dataset
from torch.utils.data import Subset, DataLoader
from utils.dataset.download_dataset import SPCUP22DatasetDownloader
from sklearn.model_selection import train_test_split


class SPCUP22DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_root: str = None,
        config_file_path: str = "config/dataset.yaml",
        dataset_name: str = "spcup22",
        annotations_file_name: str = "labels.csv",
        should_include_unseen_in_training_data: bool = False,
        should_load_eval_data: bool = False,
        val_pct: float = 0.1,
        test_pct: float = 0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.config_file_path = pathlib.Path(ROOT).joinpath(config_file_path)
        self.dataset_name = dataset_name
        self.config = load_config_file(self.config_file_path)[
            self.dataset_name
        ]
        self.dataset_root = pathlib.Path(dataset_root)

        self.should_include_unseen_in_training_data = (
            should_include_unseen_in_training_data
        )
        self.should_load_eval_data = should_load_eval_data
        self.annotations_file_name = annotations_file_name

        self.val_pct = val_pct
        self.test_pct = test_pct

        self.train_data_part1_path = self.dataset_root.joinpath(
            "training", "part1", "spcup_2022_training_part1"
        )
        self.train_data_part2_path = self.dataset_root.joinpath(
            "training", "part2", "spcup_2022_unseen"
        )
        self.evaluation_data_part1_path = self.dataset_root.joinpath(
            "evaluation", "part1", "spcup_2022_eval_part1"
        )

    @property
    def annotation_csv_filename(self):
        if (
            self.should_load_eval_data
            and self.annotations_file_name == "labels.csv"
        ):
            # try to fall back to the default csv file name if the annotation
            # filename is not passed to the constructor explicitly
            return "labels_eval_part1.csv"

        return self.annotations_file_name

    def prepare_data(self) -> None:
        downloader = SPCUP22DatasetDownloader(
            self.config_file_path, dataset_name=self.dataset_name
        )
        downloader.download_datasets()

    def read_annotations_file(self, annotation_file_path: str) -> pd.DataFrame:
        """
        Reads a annotation csv file and returns a dataframe
        """
        return pd.read_csv(annotation_file_path)

    def combine_dataframes_vertically(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Used to combine the dataframes from part 1 and part 2 of the dataset
        and return a whole dataframe that contains data from both known
        and unknown labels
        """
        return pd.concat((df1, df2), axis=0)

    def construct_full_data_paths(
        self, df: pd.DataFrame, dataset_root: pathlib.Path
    ) -> pd.DataFrame:
        """
        Modifies the `track` column of the dataset by constructing
        full paths
        """
        df["track"] = df["track"].apply(
            lambda track_name: dataset_root.joinpath(track_name)
        )
        return df

    def get_annotation_df(self, dataset_root: pathlib.Path) -> pd.DataFrame:
        csv_path = dataset_root.joinpath(self.annotation_csv_filename)
        df = self.read_annotations_file(str(csv_path))
        df = self.construct_full_data_paths(df, dataset_root)
        return df

    def get_train_val_test_split_for_training(self):
        """
        Called only during training mode by self.setup
        """
        data = self.data
        annotations_df = data.annotations_df
        total_num_samples = len(annotations_df)
        labels = annotations_df.iloc[:, 1].values

        # train and test sets
        train_indices, test_indices, _, _ = train_test_split(
            range(total_num_samples),
            labels,
            stratify=labels,
            test_size=self.test_pct,
            random_state=42,
        )

        train_labels = labels[train_indices]
        train_indices, val_indices, _, _ = train_test_split(
            train_indices,
            train_labels,
            stratify=train_labels,
            test_size=self.val_pct,
            random_state=42,
        )

        # construct subsets with the indices
        train_data = Subset(data, train_indices)
        val_data = Subset(data, val_indices)
        test_data = Subset(data, test_indices)

        return train_data, val_data, test_data

    def setup(self, stage: Optional[str] = None) -> None:
        df_part1 = self.get_annotation_df(self.train_data_part1_path)
        self.data = SPCUP22Dataset(df_part1)

        # evaluation mode, no training will be done
        if self.should_load_eval_data:
            eval_df = self.get_annotation_df(self.evaluation_data_part1_path)
            self.test_data = SPCUP22Dataset(eval_df, mode="eval")
            self.num_test_samples = self.test_data.num_samples
            return

        if self.should_include_unseen_in_training_data:
            df_part2 = self.get_annotation_df(self.train_data_part2_path)
            combined_df = self.combine_dataframes_vertically(
                df_part1, df_part2
            )
            self.data = SPCUP22Dataset(combined_df)

        self.num_classes = len(self.data.annotations_df.iloc[:, 1].unique())

        (
            self.train_data,
            self.val_data,
            self.test_data,
        ) = self.get_train_val_test_split_for_training()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=8,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
