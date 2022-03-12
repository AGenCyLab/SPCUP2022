import os
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT))

import unittest

from utils.metrics import plot_classification_report


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.save_path = ROOT.joinpath("results", "test")
        os.makedirs(self.save_path, exist_ok=True)
        self.batch_size = 100
        self.num_classes = 5
        self.file_names = ["cnf_matrix.eps", "precision_recall.eps", "roc.eps"]

    def test_metrics(self):
        actual_labels = np.random.randint(
            low=0, high=self.num_classes, size=(self.batch_size,)
        )
        predicted_probabilities = np.random.randn(
            self.batch_size, self.num_classes
        )
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        plot_classification_report(
            actual_labels,
            predicted_labels,
            predicted_probabilities,
            title_suffix="Test",
            save_path=self.save_path,
            figsize=(10, 10),
        )

        for filename in self.file_names:
            assert os.path.exists(str(self.save_path.joinpath(filename)))

    def tearDown(self) -> None:
        shutil.rmtree(str(self.save_path))
