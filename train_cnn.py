import os
import argparse
import pathlib

import torch
from utils.config import load_config_file
from datasets.SPCUP22MelDataModule import SPCUP22MelDataModule
from models.CNNs import CNNs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

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
    parser.add_argument("--checkpoint-path", default="./checkpoints", type=str)
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default="0",
        help="""A comma separated list of GPU indices. Set as value of 
        CUDA_VISIBLE_DEVICES environment variable""",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    parser.add_argument(
        "--load-eval-data", default=0, type=int,
    )
    parser.add_argument(
        "--load-augmented-data", default=False, type=bool,
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indices

    training_config_file = load_config_file(args.training_config_file_path)
    training_config = training_config_file["cnn"]

    data_module = SPCUP22MelDataModule(
        training_config["training"]["batch_size"],
        dataset_root=pathlib.Path("./data/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_load_eval_data=args.load_eval_data,
        should_include_augmented_data=args.load_augmented_data,
        num_workers=args.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    classifier = CNNs(
        args.model_type,
        learning_rate = training_config["training"]["learning_rate"],
        lr_scheduler_factor = training_config["training"]["lr_scheduler_factor"],
        lr_scheduler_patience= training_config["training"]["lr_scheduler_patience"],
    )

    cbs = [
        ModelCheckpoint(
        dirpath=args.checkpoint_path,
        every_n_epochs=training_config["training"][
            "save_checkpoint_epoch_interval"
        ],
        monitor="val_loss",
        filename=f"{args.model_type}-{{epoch}}-{{val_acc:.2f}}-{{val_loss:.2f}}",
        save_last=True,
    ),
    ]
    

    trainer = pl.Trainer(
        gpus=args.gpu_indices,
        sync_batchnorm=True,
        strategy="ddp",
        max_epochs=args.epochs,
        callbacks=cbs,
    )

    # Training on 6000 train samples which are splited into 3 datasets (train, eval, test)
    trainer.fit(
    classifier,
    datamodule=data_module,
    ckpt_path=args.resume_from_checkpoint,
    )

    trainer.test(
    classifier,
    data_module,
    ckpt_path=args.resume_from_checkpoint,
    )
