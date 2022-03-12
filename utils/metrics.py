import pathlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F


def pytorch_lightning_make_predictions(
    checkpoint: pl.LightningModule, data_module: pl.LightningDataModule
):
    """
    only to be used with pytorch lightning checkpoints and data module.

    makes predictions on the test dataloader, flattens the outputs,
    computes the predicted labels from the probabilities of each sample
    and returns the following:

    actual labels
    predicted labels
    predicted probabilities for each label per sample
    """
    checkpoint.eval()
    trainer = pl.Trainer(gpus=[4])
    predictions = trainer.predict(checkpoint, datamodule=data_module)

    flattened_predictions = []
    flattened_probabilities = []

    for batch in predictions:
        for prediction in batch:
            softmax_probabilities = F.softmax(prediction)
            predicted_label = softmax_probabilities.argmax(dim=0)
            flattened_predictions.append(predicted_label.item())
            flattened_probabilities.append(softmax_probabilities.tolist())

    actual_labels = []

    for data in data_module.test_data:
        actual_labels.append(data[-1])

    return actual_labels, flattened_predictions, flattened_probabilities


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
    figsize=(30, 10),
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
