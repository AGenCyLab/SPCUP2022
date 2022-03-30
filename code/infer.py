import argparse
import csv
from datetime import datetime
import os
import pathlib
from typing import List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.metrics import pytorch_lightning_make_predictions
from utils.config import load_config_file
from utils.audio import read_audio_file
from models.raw_audio import IncTSSDNetWrapper


class RawAudioDataset(Dataset):
    def __init__(
        self,
        wav_file_path_list: List[str],
        max_duration: float = 6.0,
    ) -> None:
        super().__init__()
        self.wav_file_path_list = wav_file_path_list
        self.max_duration = max_duration
        self.num_samples = len(self.wav_file_path_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> Tuple[np.ndarray, int, str]:
        label = -1
        filepath = self.wav_file_path_list[index]
        audio = read_audio_file(filepath, duration=self.max_duration)
        return audio, label, filepath


class RawAudioDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int, wav_file_root: str, num_workers: int = 0
    ):
        self.wav_file_root = wav_file_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        wav_file_root = pathlib.Path(self.wav_file_root)
        wav_files_list = list(wav_file_root.glob("*.wav"))
        wav_files_list = list(
            map(lambda filepath: str(filepath), wav_files_list)
        )
        self.wav_files_list = wav_files_list

    def setup(self, stage: Optional[str] = None) -> None:
        self.test_data = RawAudioDataset(self.wav_files_list)
        self.num_test_samples = self.test_data.num_samples

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The path to the directory containing the wav files to be used for inference",
    )
    parser.add_argument(
        "--training-config-file-path",
        default="config/train_params.yaml",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint-path",
        default="./checkpoints/tssdnet/inc_tssdnet_with_unseen_aug/last.ckpt",
        type=str,
        help="The path to the checkpoint to use for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of processes to use for data loading",
    )
    parser.add_argument(
        "--answer-path",
        type=str,
        default="./scores",
        help="""The path in which the answer csv file will be placed""",
    )

    return parser


def write_score(
    predictions: List[int], filepaths: List[str], answer_path: pathlib.Path
):
    data = [
        [pathlib.Path(filepath).name, prediction]
        for filepath, prediction in zip(filepaths, predictions)
    ]

    with open(
        answer_path.joinpath("score.csv"), "w", encoding="UTF8", newline=""
    ) as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(data)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name = "inc_tssdnet"
    feature_name = "raw_audio"
    current_timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    training_config_file = load_config_file(args.training_config_file_path)
    training_config = training_config_file["tssdnet"]

    data_module = RawAudioDataModule(
        training_config["training"]["batch_size"],
        args.dataset_path,
        num_workers=args.num_workers,
    )

    data_module.prepare_data()
    data_module.setup()

    classifier = IncTSSDNetWrapper.load_from_checkpoint(
        args.model_checkpoint_path
    )

    answer_path = pathlib.Path(args.answer_path).joinpath(
        "{}-{}-{}".format(model_name, feature_name, current_timestamp)
    )
    os.makedirs(answer_path, exist_ok=True)

    (
        actual_labels,
        flattened_predictions,
        flattened_probabilities,
        filepaths,
    ) = pytorch_lightning_make_predictions(
        classifier, data_module, mode="eval"
    )

    write_score(flattened_predictions, filepaths, answer_path)
