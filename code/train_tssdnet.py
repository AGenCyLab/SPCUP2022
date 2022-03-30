import os
import argparse
import pathlib

import torch
from utils.config import load_config_file
from datasets.SPCUP22DataModule import SPCUP22DataModule
from models.raw_audio import IncTSSDNetWrapper
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-config-file-path",
        default="config/dataset.yaml",
        type=str,
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
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--include-augmented-data", action="store_true", default=False
    )
    parser.add_argument(
        "--include-unseen-in-training-data", action="store_true", default=False
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indices

    training_config_file = load_config_file(args.training_config_file_path)
    training_config = training_config_file["tssdnet"]

    data_module = SPCUP22DataModule(
        training_config["training"]["batch_size"],
        dataset_root=pathlib.Path("./data/raw_audio/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_include_augmented_data=args.include_augmented_data,
        should_include_unseen_in_training_data=args.include_unseen_in_training_data,
        should_load_eval_data=False,
    )
    data_module.prepare_data()
    data_module.setup()

    print("Number of classes:", data_module.num_classes)
    print("Number of samples:", len(data_module.data.annotations_df))

    classifier = IncTSSDNetWrapper(
        num_classes=data_module.num_classes,
        learning_rate=training_config["training"]["learning_rate"],
        exp_lr_scheduler_gamma=training_config["training"][
            "exp_lr_scheduler_gamma"
        ],
    )

    classifier.train()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        every_n_epochs=training_config["training"][
            "save_checkpoint_epoch_interval"
        ],
        monitor="val_loss",
        save_last=True,
    )

    all_callbacks = [checkpoint_callback]

    try:
        wandb_project = training_config["wandb"]["project"]
        wandb_entity = training_config["wandb"]["entity"]
    except Exception as exception:
        print(exception)
        wandb_project = ""
        wandb_entity = ""

    wandb_logger = None
    if wandb_project != "" and wandb_entity != "":
        wandb_logger = WandbLogger(project=wandb_project, entity=wandb_entity)
        lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
        all_callbacks.append(lr_monitor_callback)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        logger=wandb_logger,
        max_epochs=args.epochs,
        sync_batchnorm=True,
        accelerator="ddp",
        callbacks=all_callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.fit(classifier, datamodule=data_module)

    data_module.teardown()
