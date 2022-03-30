import pathlib
import sys
from typing import Callable, Optional, Tuple

import pandas as pd

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from utils.config import load_config_file
from utils.dataset import SPCUP22DatasetDownloader

from datasets.SPCUP22Dataset import SPCUP22Dataset


class SPCUP22DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_type: str = "raw_audio",
        dataset_root: str = None,
        config_file_path: str = "config/dataset.yaml",
        dataset_name: str = "spcup22",
        should_include_augmented_data: bool = False,
        should_include_unseen_in_training_data: bool = False,
        should_load_eval_data: bool = False,
        val_pct: float = 0.1,
        test_pct: float = 0.2,
        transform: Optional[Callable] = None,
        num_workers: int = 8,
    ):
        """
        data_type is one of ("raw_audio", "mel_features", ...) and any other
        type of precomputed data that we might want to use. The dataset.yaml
        file should be updated accordingly with the proper key.
        """
        super().__init__()
        self.batch_size = batch_size
        self.config_file_path = pathlib.Path(ROOT).joinpath(config_file_path)
        self.dataset_name = dataset_name
        self.config = load_config_file(self.config_file_path)[
            self.dataset_name
        ]
        self.dataset_root = pathlib.Path(dataset_root)
        self.transform = transform
        self.data_type = data_type

        self.should_include_unseen_in_training_data = (
            should_include_unseen_in_training_data
        )
        self.should_include_augmented_data = should_include_augmented_data
        self.should_load_eval_data = should_load_eval_data

        self.val_pct = val_pct
        self.test_pct = test_pct
        self.num_workers = num_workers

        self.train_data_part1_path = self.dataset_root.joinpath(
            "training", "part1", "spcup_2022_training_part1"
        )
        self.train_data_part2_path = self.dataset_root.joinpath(
            "training", "part2", "spcup_2022_unseen"
        )
        self.train_data_part1_aug_path = self.dataset_root.joinpath(
            "training",
            "part1_aug",
            "part1_aug",
            "spcup_2022_training_part1",
        )
        self.train_data_part2_aug_path = self.dataset_root.joinpath(
            "training",
            "part2_aug",
            "part2_aug",
            "spcup_2022_unseen",
        )
        self.evaluation_data_part1_path = self.dataset_root.joinpath(
            "evaluation", "part1", "spcup_2022_eval_part1"
        )
        self.evaluation_data_part2_path = self.dataset_root.joinpath(
            "evaluation", "part2", "spcup_2022_eval_part2"
        )
        self.annotation_file_map = {
            "training_part1": "labels.csv",
            "training_part1_aug": "labels_aug.csv",
            "training_part2": "labels.csv",
            "training_part2_aug": "labels_aug.csv",
            "evaluation_part1": "labels_eval_part1.csv",
            "evaluation_part2": "labels_eval_part2.csv",
        }

    def get_default_annotation_csv_filename(self, key: str):
        return self.annotation_file_map[key]

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

    def combine_dataframes_vertically(
        self, dfs: Tuple[pd.DataFrame, ...]
    ) -> pd.DataFrame:
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

    def get_annotation_df(
        self, dataset_root: pathlib.Path, key: str = None
    ) -> pd.DataFrame:
        if key is None:
            raise Exception(
                """Need a valid key among the following to get CSV filename:
                    'training_part1'
                    'training_part1_aug'
                    'training_part2'
                    'training_part2_aug'
                    'evaluation_part1'
                    'evaluation_part2'
                """
            )

        csv_path = dataset_root.joinpath(
            self.get_default_annotation_csv_filename(key)
        )
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

        self.num_train_samples = len(train_indices)
        self.num_val_samples = len(val_indices)
        self.num_test_samples = len(test_indices)

        # construct subsets with the indices
        train_data = Subset(data, train_indices)
        val_data = Subset(data, val_indices)
        test_data = Subset(data, test_indices)

        return train_data, val_data, test_data

    def setup(self, stage: Optional[str] = None) -> None:
        # evaluation mode, no training will be done
        if self.should_load_eval_data:
            if self.should_include_augmented_data:
                eval_df = self.get_annotation_df(
                    self.evaluation_data_part2_path, key="evaluation_part2"
                )
            else:
                eval_df = self.get_annotation_df(
                    self.evaluation_data_part1_path, key="evaluation_part1"
                )

            self.test_data = SPCUP22Dataset(
                eval_df, mode="eval", transform=self.transform
            )
            self.num_test_samples = self.test_data.num_samples
            return

        # training data
        # without unseen + augmented
        df_part1 = self.get_annotation_df(
            self.train_data_part1_path, key="training_part1"
        )
        combined_df = df_part1

        # augmented without unseen class
        if self.should_include_augmented_data:
            df_aug_part1 = self.get_annotation_df(
                self.train_data_part1_aug_path, key="training_part1_aug"
            )
            combined_df = self.combine_dataframes_vertically(
                (
                    combined_df,
                    df_aug_part1,
                )
            )

        if self.should_include_unseen_in_training_data:
            # unseen class without augmented
            df_part2 = self.get_annotation_df(
                self.train_data_part2_path, key="training_part2"
            )
            combined_df = self.combine_dataframes_vertically(
                (
                    combined_df,
                    df_part2,
                )
            )

            # augmented + unseen
            if self.should_include_augmented_data:
                df_aug_part2 = self.get_annotation_df(
                    self.train_data_part2_aug_path, key="training_part2_aug"
                )
                combined_df = self.combine_dataframes_vertically(
                    (
                        combined_df,
                        df_aug_part2,
                    )
                )

        self.data = SPCUP22Dataset(combined_df, transform=self.transform)
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
