import pathlib
import sys

ROOT = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT))

import unittest

from datasets.SPCUP22DataModule import SPCUP22DataModule
from features.audio import MFCC


class TestFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.BATCH_SIZE = 16
        self.datamodule = SPCUP22DataModule(
            self.BATCH_SIZE, str(ROOT.joinpath("data", "spcup22"))
        )
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def test_mfcc(self):
        assert hasattr(self.datamodule, "train_data")
        duration = 6

        sample, label = self.datamodule.data.__getitem__(0)
        sample_rate = int(sample.shape[-1] / duration)

        n_mfcc_expected = 40
        hop_length = 256

        # the audios are 16KHz
        self.assertEqual(sample_rate, 16000)

        mfcc = MFCC(
            sr=sample_rate, n_mfcc=n_mfcc_expected, hop_length=hop_length
        )
        mfcc_features, label = mfcc(
            (
                sample,
                label,
            )
        )

        num_samples, mfcc_count, num_features = mfcc_features.shape
        self.assertEqual(num_samples, 1)
        self.assertEqual(mfcc_count, n_mfcc_expected)

    def tearDown(self) -> None:
        self.datamodule.teardown()
