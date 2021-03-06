import pathlib
import sys
from typing import Optional

import pandas as pd

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from utils.config import load_config_file
from utils.dataset import SPCUP22DatasetDownloader

from datasets.SPCUP22MelDataset import SPCUP22MelDataset


class SPCUP22MelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_root: str = None,
        config_file_path: str = "config/dataset.yaml",
        dataset_name: str = "spcup22",
        annotations_file_name: str = "labels.csv",
        should_load_eval_data: int = 0,  # 0 = load train data, 1 = load eval part 1 data, 2 = load eval part 2
        should_include_augmented_data: bool = False,
        data_type: str = "mel_feature",
        val_pct: float = 0.1,
        test_pct: float = 0.2,
        num_workers=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.config_file_path = pathlib.Path(ROOT).joinpath(config_file_path)
        self.dataset_name = dataset_name
        self.config = load_config_file(self.config_file_path)[
            self.dataset_name
        ]
        self.dataset_root = pathlib.Path(dataset_root)
        self.data_type = data_type

        self.should_load_eval_data = should_load_eval_data
        self.should_include_augmented_data = should_include_augmented_data
        self.annotations_file_name = annotations_file_name

        self.val_pct = val_pct
        self.test_pct = test_pct

        self.num_workers = num_workers

        self.train_data_path = self.dataset_root.joinpath(
            "training",
        )
        self.train_augmented_data_path = self.dataset_root.joinpath(
            "training_augmented",
        )
        self.evaluation_data_part1_path = self.dataset_root.joinpath(
            "evaluation_part1",
        )
        self.evaluation_data_part2_path = self.dataset_root.joinpath(
            "evaluation_part2",
        )

    @property
    def annotation_csv_filename(self):
        if (
            self.should_load_eval_data
            and self.annotations_file_name == "labels.csv"
        ):
            # try to fall back to the default csv file name if the annotation
            # filename is not passed to the constructor explicitly
            return "labels.csv"

        return self.annotations_file_name

    def prepare_data(self) -> None:
        downloader = SPCUP22DatasetDownloader(
            self.config_file_path,
            dataset_name=self.dataset_name,
            data_type=self.data_type,
        )
        downloader.download_datasets()

    def read_annotations_file(self, annotation_file_path: str) -> pd.DataFrame:
        """
        Reads a annotation csv file and returns a dataframe
        """
        return pd.read_csv(annotation_file_path)

    def construct_full_data_paths(
        self, df: pd.DataFrame, dataset_root: pathlib.Path
    ) -> pd.DataFrame:
        """
        Modifies the `track` column of the dataset by constructing
        full paths
        """
        df["track"] = df["track"].apply(
            lambda track_name: str(dataset_root.joinpath(track_name))
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

    def combine_dataframes_vertically(self, dfs):
        """
        Used to combine the dataframes. Useful for concatenating unseen data
        and augmented data together for training
        Args:
            dfs: A tuple of dataframes. The column names should match for each
            dataframe in the tuple
        """
        for df in dfs:
            df.columns = df.columns.str.strip()
            df.reset_index(drop=True, inplace=True)

        result = pd.concat(dfs, ignore_index=True)

        return result

    def setup(self, stage: Optional[str] = None) -> None:
        # evaluation mode, no training will be done
        if self.should_load_eval_data == 1:
            eval_df = self.get_annotation_df(self.evaluation_data_part1_path)
            self.test_data = SPCUP22MelDataset(
                eval_df,
                mode="eval",
            )
            self.num_test_samples = len(self.test_data)
            return
        if self.should_load_eval_data == 2:
            eval_df = self.get_annotation_df(self.evaluation_data_part2_path)
            self.test_data = SPCUP22MelDataset(
                eval_df,
                mode="eval",
            )
            self.num_test_samples = len(self.test_data)
            return

        train_df = self.get_annotation_df(self.train_data_path)

        if self.should_include_augmented_data:
            train_augmented_df = self.get_annotation_df(
                self.train_augmented_data_path
            )
            train_df = self.combine_dataframes_vertically(
                [train_df, train_augmented_df]
            )

        self.data = SPCUP22MelDataset(train_df)

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
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
