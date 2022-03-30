import pathlib
import sys

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

from typing import Optional
import pandas as pd
from datasets.SPCUP22DataModule import SPCUP22DataModule
from datasets.SPCUP22MelDataset import SPCUP22MelDataset


class SPCUP22MelDataModule(SPCUP22DataModule):
    def __init__(
        self,
        batch_size: int,
        data_type: str = "mel_feature",
        dataset_root: str = None,
        config_file_path: str = "config/mel_feature.yaml",
        dataset_name: str = "spcup22",
        should_include_augmented_data: bool = False,
        should_include_unseen_in_training_data: bool = False,
        # 0 = don't load eval data, 1 = load eval part 1, 2 = load eval part 2
        should_load_eval_data: int = 0,
        val_pct: float = 0.1,
        test_pct: float = 0.2,
        num_workers: int = 0,
        annotations_file_name: str = "labels.csv",
    ):
        super().__init__(
            batch_size,
            data_type=data_type,
            dataset_root=dataset_root,
            config_file_path=config_file_path,
            dataset_name=dataset_name,
            should_include_augmented_data=should_include_augmented_data,
            should_include_unseen_in_training_data=should_include_unseen_in_training_data,
            should_load_eval_data=False,
            val_pct=val_pct,
            test_pct=test_pct,
            transform=None,
            num_workers=num_workers,
        )

        self.should_load_eval_data = should_load_eval_data
        self.annotations_file_name = annotations_file_name
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

    def get_default_annotation_csv_filename(self):
        return self.annotations_file_name

    def get_annotation_df(self, dataset_root: pathlib.Path) -> pd.DataFrame:
        csv_path = dataset_root.joinpath(
            self.get_default_annotation_csv_filename()
        )
        df = self.read_annotations_file(str(csv_path))
        df = self.construct_full_data_paths(df, dataset_root)
        return df

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
