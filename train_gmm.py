import os
from argparse import ArgumentParser
import pathlib
import numpy as np
from pycave.bayes.gmm.estimator import GaussianMixture
import torch
from utils.dataset import get_numpy_dataset_from_dataloader
from utils.config import load_config_file
from datasets.SPCUP22DataModule import SPCUP22DataModule
from features.audio import MFCC
from torchvision.transforms import Compose
from tqdm import tqdm


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

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
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default="0",
        help="""A comma separated list of GPU indices. Set as value of 
        CUDA_VISIBLE_DEVICES environment variable""",
    )
    parser.add_argument(
        "--checkpoint-save-path", type=str, default="./checkpoints/gmm"
    )
    parser.add_argument(
        "--include-augmented-data", action="store_true", default=False
    )

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

    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    training_config_file = load_config_file(args.training_config_file_path)
    train_config = training_config_file["gmm"]

    batch_size = train_config["training"]["batch_size"]
    epochs = train_config["training"]["epochs"]
    n_mfcc = train_config["features"]["n_mfcc"]
    hop_length = train_config["features"]["hop_length"]

    # feature
    mfcc_extractor = MFCC(n_mfcc=n_mfcc, hop_length=hop_length)
    transforms = Compose([mfcc_extractor])

    data_module = SPCUP22DataModule(
        batch_size,
        dataset_root=pathlib.Path("./data/raw_audio/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_include_augmented_data=args.include_augmented_data,
        should_include_unseen_in_training_data=args.include_unseen_in_training_data,
        should_load_eval_data=args.load_eval_data,
        transform=transforms,
    )
    data_module.prepare_data()
    data_module.setup()

    print("Number of classes:", data_module.num_classes)
    print("Number of samples:", len(data_module.data.annotations_df))

    train_data = data_module.train_dataloader()
    X = get_numpy_dataset_from_dataloader(train_data, batch_size)

    estimator = GaussianMixture(
        num_components=data_module.num_classes,
        covariance_regularization=1e-6,
        batch_size=batch_size,
        trainer_params={
            "accelerator": "gpu",
            "max_epochs": epochs,
            "gpus": torch.cuda.device_count(),
        },
    )
    estimator.fit(X)

    print("Means: ")
    print(estimator.model_.means)

    print("Component Probs: ")
    print(estimator.model_.component_probs)

    print("Precisions Cholesky: ")
    print(estimator.model_.precisions_cholesky)

    estimator.save(args.checkpoint_save_path)
