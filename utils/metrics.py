from typing import Union, List
import pathlib
from zipfile import ZipFile
from sklearn.metrics import accuracy_score, f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


def pytorch_lightning_make_predictions(
    checkpoint: pl.LightningModule,
    data_module: pl.LightningDataModule,
    mode: str = "training",
):
    """
    only to be used with pytorch lightning checkpoints and data module.

    makes predictions on the test dataloader, flattens the outputs,
    computes the predicted labels from the probabilities of each sample
    and returns the following:

    actual labels (an empty list if mode = "eval")
    predicted labels
    predicted probabilities for each label per sample
    filepaths

    Args:
        mode: one of ["training", "eval"]
        By training, it is implied that the predictions are being carried out
        on a subset of heldout data from the training set and not on the actual
        eval set. Since the eval set has no labels, it's not possible to have
        actual labels.
    """
    checkpoint.eval()
    trainer = pl.Trainer()#gpus=torch.cuda.device_count(), accelerator="ddp")
    predictions = trainer.predict(checkpoint, datamodule=data_module)

    filepaths = []
    flattened_predictions = []
    flattened_probabilities = []

    for batch in predictions:
        current_predictions, current_filepaths = batch

        for prediction, filepath in zip(
            current_predictions, current_filepaths
        ):
            filepaths.append(filepath)
            softmax_probabilities = F.softmax(prediction)
            predicted_label = softmax_probabilities.argmax(dim=0)
            flattened_predictions.append(predicted_label.item())
            flattened_probabilities.append(softmax_probabilities.tolist())

    actual_labels = []

    if mode == "training":
        for data in data_module.test_data:
            actual_labels.append(data[-1])

    return (
        actual_labels,
        flattened_predictions,
        flattened_probabilities,
        filepaths,
    )


def print_scores(
    actual_labels, predicted_labels,
):
    """
    prints out f1 score (micro average) and accuracy scores. useful
    for debugging or simply reporting accuracy quickly
    """
    f1 = f1_score(actual_labels, predicted_labels, average="micro")
    accuracy = accuracy_score(actual_labels, predicted_labels)

    print(
        """
    F1 Score: {:.2f}
    Accuracy: {:.2f}
    """.format(
            f1, accuracy
        )
    )


def plot_classification_report(
    actual_labels,
    predicted_labels,
    predicted_probabilities,
    title_suffix,
    save_path,
    figsize=(10, 10),
    dpi=300,
):
    """
    plots the confusion matrix, precision recall curves and roc curves
    for a given set of labels and predictions.
    """
    root = pathlib.Path(save_path)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    plt.tight_layout()

    skplt.metrics.plot_confusion_matrix(
        actual_labels,
        predicted_labels,
        title="Confusion Matrix: {}".format(title_suffix),
        ax=ax,
    )

    fig.savefig(root.joinpath("cnf_matrix.eps"), format="eps")

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    plt.tight_layout()

    skplt.metrics.plot_precision_recall(
        actual_labels,
        predicted_probabilities,
        title="Precision-Recall Curve: {}".format(title_suffix),
        figsize=(6, 6),
        ax=ax,
    )

    fig.savefig(root.joinpath("precision_recall.eps"), format="eps")

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    plt.tight_layout()

    skplt.metrics.plot_roc(
        actual_labels,
        predicted_probabilities,
        title="ROC Curve: {}".format(title_suffix),
        figsize=(6, 6),
        ax=ax,
    )

    fig.savefig(root.joinpath("roc.eps"), format="eps")


def write_answers(
    submission_path: Union[str, pathlib.Path],
    flattened_predictions: List[int],
    filepaths: List[str],
):
    """
    Creates the answer.txt and answer.zip file following the codalab conventions

    Args:
        submission_path: either a str or a pathlib.Path object with the path
        to the folder where the answers should be stored
    
        flattened_predictions: a 1D list of predictions

        filepaths: a 1D list of filepaths. The indices should correspond to the
        predictions

    Returns: 
        None
    """
    answers = []
    answer_txt_file_path = pathlib.Path(submission_path).joinpath(
        "answer.txt",
    )

    for prediction, filepath in zip(flattened_predictions, filepaths):
        filename = pathlib.Path(filepath).name
        answer = "{}, {}\n".format(filename, prediction)
        answers.append(answer)

    with open(answer_txt_file_path, "w") as answer_text_file_object:
        answer_text_file_object.writelines(answers)

    zipObj = ZipFile(str(submission_path.joinpath("answer.zip")), "w")
    zipObj.write(str(answer_txt_file_path), "answer.txt")
    zipObj.close()
