from datetime import datetime
import os
from argparse import ArgumentParser
import pathlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from utils.config import load_config_file
from datasets.SPCUP22DataModule import SPCUP22DataModule
from features.audio import MFCC
from torchvision.transforms import Compose
import pickle


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


if __name__ == "__main__":
    current_timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    # config and params
    train_config = load_config_file("./config/train_params.yaml")["svm"]

    batch_size = train_config["training"]["batch_size"]
    epochs = train_config["training"]["epochs"]
    n_mfcc = train_config["features"]["n_mfcc"]
    hop_length = train_config["features"]["hop_length"]

    # feature
    mfcc_extractor = MFCC(n_mfcc=n_mfcc, hop_length=hop_length)
    transforms = Compose([mfcc_extractor])

    # datamodule
    data_module = SPCUP22DataModule(
        batch_size=batch_size,
        dataset_root=str(pathlib.Path("./data/spcup22").absolute()),
        transform=transforms,
    )
    data_module.prepare_data()
    data_module.setup()
    classes = np.array(list(range(data_module.num_classes)), dtype=int)

    # svm
    classifier = SGDClassifier(**train_config["params"])

    # callbacks

    # others ...
    validation_error = float("inf")

    # fit
    for epoch in tqdm(range(epochs)):
        train_data = data_module.train_dataloader()
        val_data = data_module.val_dataloader()

        for batch in train_data:
            samples, labels = batch
            samples = np.reshape(samples, newshape=(batch_size, -1))
            classifier.partial_fit(samples, labels, classes=classes)

        current_val_error = 0
        num_val_batches = 0

        for batch in val_data:
            samples, labels = batch
            samples = np.reshape(samples, newshape=(batch_size, -1))
            accuracy = classifier.score(samples, labels)

            error = 1 - accuracy

            current_val_error += error
            num_val_batches += 1

        current_val_error /= num_val_batches

        print("Validation Error: {:.2f}".format(current_val_error))

        # only save the checkpoint which has less validation error
        if current_val_error < validation_error:
            validation_error = current_val_error
            model_filename = "svm-{}-{:.2f}.pkl".format(
                current_timestamp, current_val_error
            )
            model_path = str(
                pathlib.Path(args.checkpoint_save_path).joinpath(
                    model_filename
                )
            )
            save_checkpoint(classifier, model_path, train_config["params"])

    # save the last checkpoint after all epochs are completed
    model_filename = "svm-{}-{:.2f}.pkl".format("last", current_val_error)
    model_path = str(
        pathlib.Path(args.checkpoint_save_path).joinpath(model_filename)
    )
    save_checkpoint(classifier, model_path, train_config["params"])

