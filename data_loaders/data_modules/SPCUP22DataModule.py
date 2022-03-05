import pathlib
import sys
from typing import Optional

ROOT = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
import torch
import yaml
from data_loaders.datasets.SPCUP22Dataset import SPCUP22Dataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from utils.dataset.download_dataset import SPCUP22DatasetDownloader


class SPCUP22DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_root: str = None,
        config_file_path: str = "config/dataset.yaml",
        dataset_name: str = "spcup22",
        should_include_unseen_in_training_data: bool = False,
        should_load_eval_data: bool = False,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.config_file_path = pathlib.Path(ROOT).joinpath(config_file_path)
        self.dataset_name = dataset_name
        self.config = self.load_config_file(self.config_file_path)[
            self.dataset_name
        ]
        self.dataset_root = pathlib.Path(dataset_root)

        self.should_include_unseen_in_training_data = (
            should_include_unseen_in_training_data
        )
        self.should_load_eval_data = should_load_eval_data

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

    def load_config_file(self, config_file_path):
        with open(config_file_path, mode="r") as config_file_object:
            config = yaml.load(config_file_object, Loader=yaml.FullLoader)
            return config

    def prepare_data(self) -> None:
        downloader = SPCUP22DatasetDownloader(
            self.config_file_path, dataset_name=self.dataset_name
        )
        downloader.download_datasets()

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = SPCUP22Dataset(str(self.train_data_part1_path))

        if self.should_include_unseen_in_training_data:
            unseen_data = SPCUP22Dataset(str(self.train_data_part2_path))
            self.data = ConcatDataset([self.data, unseen_data])

        self.total_samples = self.data.num_samples

        self.num_test_samples = int(self.total_samples * self.test_pct)
        self.num_train_samples = self.total_samples - self.num_test_samples
        self.num_val_samples = int(self.num_train_samples * self.val_pct)
        self.num_train_samples = self.num_train_samples - self.num_val_samples

        self.train_data, self.val_data, self.test_data = random_split(
            self.data,
            [
                self.num_train_samples,
                self.num_val_samples,
                self.num_test_samples,
            ],
            generator=torch.Generator().manual_seed(42),
        )

        if self.should_load_eval_data:
            self.test_data = SPCUP22Dataset(
                str(self.evaluation_data_part1_path), mode="eval"
            )
            self.num_test_samples = self.test_data.num_samples

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
