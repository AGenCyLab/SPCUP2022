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
        default="VGG16",
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
        should_load_eval_data=True,
        num_workers=0,
    )
    data_module.prepare_data()
    data_module.setup()

    classifier = CNNs(
        network=args.model_type,
        num_classes=5,
        learning_rate = training_config["training"]["learning_rate"],
        lr_scheduler_factor = training_config["training"]["lr_scheduler_factor"],
        lr_scheduler_patience= training_config["training"]["lr_scheduler_patience"],
    )

    classifier.test()


    trainer = Trainer(
        gpus=torch.cuda.device_count(),
    )

    trainer.test(
        classifier,
        data_module,
        ckpt_path=args.model_checkpoint_path,
    )