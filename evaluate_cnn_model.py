import os
import argparse
import pathlib
from datetime import datetime
import torch
from utils.config import load_config_file
from datasets.SPCUP22MelDataModule import SPCUP22MelDataModule
from models.CNNs import CNNs
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from utils.metrics import pytorch_lightning_make_predictions, write_answers

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["VGG16", "ResNet18", "ResNet34"],
    )
    parser.add_argument(
        "--dataset-config-file-path", default="config/mel_feature.yaml", type=str,
    )
    parser.add_argument(
        "--training-config-file-path",
        default="config/train_params.yaml",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint-path", default="./checkpoints", type=str
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="./submissions",
        help="""The path in which the submission text file will be placed""",
    )
    parser.add_argument(
        "--load-eval-data",
        type=int,
        default=0,
        
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    training_config_file = load_config_file(args.training_config_file_path)
    training_config = training_config_file["cnn"]

    data_module = SPCUP22MelDataModule(
        training_config["training"]["batch_size"],
        dataset_root=pathlib.Path("./data/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_load_eval_data=args.load_eval_data,
        num_workers=0,
    )
    data_module.prepare_data()
    data_module.setup()
    
    classifier = CNNs(
        args.model_type,
        learning_rate = training_config["training"]["learning_rate"],
        lr_scheduler_factor = training_config["training"]["lr_scheduler_factor"],
        lr_scheduler_patience= training_config["training"]["lr_scheduler_patience"],
    ).load_from_checkpoint(
        args.model_checkpoint_path,
    )
    
    if classifier is None:
        raise Exception("Invalid model_type '{}'".format(args.model_type))
    
    feature_name = "mel_spectrogram"
    model_name = args.model_type.replace("-", "_")
    feature_name = ""
    current_timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    submission_path = pathlib.Path(args.submission_path).joinpath(
        "{}-{}-{}".format(model_name, feature_name, current_timestamp)
    )
    os.makedirs(submission_path, exist_ok=True)

    (
        actual_labels,
        flattened_predictions,
        flattened_probabilities,
        filepaths,
    ) = pytorch_lightning_make_predictions(
        classifier, data_module, mode="eval"
    )

    write_answers(submission_path, flattened_predictions, filepaths)