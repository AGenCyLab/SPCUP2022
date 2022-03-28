import argparse
from datetime import datetime
import os
import pathlib
from zipfile import ZipFile
from utils.metrics import pytorch_lightning_make_predictions, write_answers
from models.raw_audio import IncTSSDNetWrapper, ResTSSDNetWrapper
from utils.config import load_config_file
from datasets.SPCUP22DataModule import SPCUP22DataModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["res-tssdnet", "inc-tssdnet"],
        default="res-tssdnet",
    )
    parser.add_argument(
        "--dataset-config-file-path", default="config/dataset.yaml", type=str,
    )
    parser.add_argument(
        "--training-config-file-path",
        default="config/train_params.yaml",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint-path", default="./checkpoints", type=str
    )
    parser.add_argument("--include-augmented-data", action="store_true")
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

    # WARNING: multi gpu inference causes duplicate answers.txt to be produced
    # in n_gpus folders under the submission folder. Using only one GPU is better
    # in this case
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    model_name = args.model_type.replace("-", "_")
    feature_name = ""
    current_timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    training_config_file = load_config_file(args.training_config_file_path)
    training_config = training_config_file["tssdnet"]

    data_module = SPCUP22DataModule(
        training_config["training"]["batch_size"],
        dataset_root=pathlib.Path("./data/raw_audio/spcup22").absolute(),
        config_file_path=args.dataset_config_file_path,
        should_load_eval_data=True,
        should_include_augmented_data=args.include_augmented_data,
    )
    data_module.prepare_data()
    data_module.setup()

    classifier = None
    if args.model_type == "res-tssdnet":
        classifier = ResTSSDNetWrapper.load_from_checkpoint(
            args.model_checkpoint_path
        )
        feature_name = "raw_audio"
    elif args.model_type == "inc-tssdnet":
        classifier = IncTSSDNetWrapper.load_from_checkpoint(
            args.model_checkpoint_path
        )
        feature_name = "raw_audio"

    if classifier is None:
        raise Exception("Invalid model_type '{}'".format(args.model_type))

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

