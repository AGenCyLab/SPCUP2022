import os
import argparse
import pathlib

import torch
from utils.config import load_config_file
from datasets.SPCUP22MelDataModule import SPCUP22MelDataModule
from models.simpleNet import SimpleNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["simpleNet", "VGG16"],
        default="simpleNet",
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
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    train_or_eval = parser.add_mutually_exclusive_group()
    train_or_eval.add_argument(
        "--include-unseen-in-training-data", action="store_true", default=False
    )
    train_or_eval.add_argument(
        "--load-eval-data", action="store_true", default=False
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
        num_workers=args.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    if args.model_type == "simpleNet":
        classifier = SimpleNet(
            num_classes=data_module.num_classes,
            learning_rate = training_config["training"]["learning_rate"],
            lr_scheduler_factor = training_config["training"]["lr_scheduler_factor"],
            lr_scheduler_patience= training_config["training"]["lr_scheduler_patience"],
        )
    else:
        raise Exception("Other models are not yet implemented")

    classifier.train()

    cbs = [
        ModelCheckpoint(
        dirpath=args.checkpoint_path,
        save_on_train_epoch_end=True,
        every_n_epochs=training_config["training"][
            "save_checkpoint_epoch_interval"
        ],
        monitor="val_loss",
        save_last=True,
        filename=f"{args.model_type}-{{v_num:02d}}-{{epoch:02d}}-{{train_loss:.2f}}-{{val_loss:.2f}}",
        save_top_k=3,
    ),
    ]

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        sync_batchnorm=True,
        strategy="ddp",
        auto_scale_batch_size=True,
        callbacks=cbs,
        check_val_every_n_epoch=training_config["training"]["check_val_every_n_epoch"],
    )

    trainer.fit(
    classifier,
    datamodule=data_module,
    ckpt_path=args.resume_from_checkpoint
    )

    data_module.teardown()

