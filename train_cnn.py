from datetime import datetime
import os
from argparse import ArgumentParser
import pathlib
import pickle

import numpy as np
from tqdm import tqdm

from torchvision.transforms import Compose

from utils.config import load_config_file
from datasets.SPCUP22DataModule import SPCUP22DataModule
from features.audio import MFCC

def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-save-path", type=str, default="./checkpoints/svm"
    )
    return parser

def save_checkpoint(classifier, filepath, model_params=None):
    """Saves the checkpoint"""
    # add the model hyperparameters to the classifier object
    if model_params:
        classifier.model_params = model_params

    with open(filepath, "wb") as model_file_obj:
        pickle.dump(classifier, model_file_obj)

if __name__ == '__main__':
    current_timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    # config and params
    train_config = load_config_file("./config/train_params.yaml")["cnn"]

    batch_size = train_config["training"]["batch_size"]
    epochs = train_config["training"]["epochs"]
    n_mfcc = train_config["features"]["n_mfcc"]
    hop_length = train_config["features"]["hop_length"]

    # feature
    mfcc_extractor = MFCC(n_mfcc=n_mfcc, hop_length=hop_length)
    transforms = Compose([mfcc_extractor])

    # Data module
    data_module = SPCUP22DataModule(
        batch_size=batch_size,
        dataset_root=str(pathlib.Path("./data/spcup22").absolute()),
        transform=transforms,
    )
    data_module.prepare_data()
    data_module.setup()
    classes = np.array(list(range(data_module.num_classes)), dtype=int)


