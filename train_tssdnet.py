import argparse
import pathlib
import torch
from data_loaders.data_modules.SPCUP22DataModule import SPCUP22DataModule
from models.raw_audio.ResTSSDNet import ResTSSDNetWrapper
from models.raw_audio.IncTSSDNet import IncTSSDNetWrapper
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["res-tssdnet", "inc-tssdnet"],
        default="res-tssdnet",
    )
    parser.add_argument(
        "--include-unseen-in-training-data", action="store_true", default=False
    )
    parser.add_argument("--load-eval-data", action="store_true", default=False)
    parser.add_argument(
        "--dataset-config-file-path", default="config/dataset.yaml", type=str,
    )
    parser.add_argument("--checkpoint-path", default="./checkpoints", type=str)
    parser.add_argument(
        "--gpus", type=str, default="0",
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--data-module-mode",
        type=str,
        default="training",
        choices=["training", "eval"],
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    data_module = SPCUP22DataModule(
        args.batch_size,
        dataset_root=pathlib.Path("./data/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_include_unseen_in_training_data=args.include_unseen_in_training_data,
        should_load_eval_data=args.load_eval_data,
    )

    if args.model_type == "res-tssdnet":
        classifier = ResTSSDNetWrapper()
    elif args.model_type == "inc-tssdnet":
        classifier = IncTSSDNetWrapper()

    classifier.train()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        every_n_val_epochs=1,
        monitor="val_loss",
        save_last=True,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        # logger=wandb_logger,
        max_epochs=args.epochs,
        sync_batchnorm=True,
        accelerator="ddp",
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.fit(classifier, datamodule=data_module)

