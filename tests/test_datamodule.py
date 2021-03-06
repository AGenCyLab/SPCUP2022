import pathlib
import sys

ROOT = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT))

import unittest

from datasets.SPCUP22DataModule import SPCUP22DataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        self.BATCH_SIZE = 16
        self.NUM_SAMPLES_WITH_UNSEEN = 6000
        self.NUM_SAMPLES_WITHOUT_UNSEEN = 5000
        self.NUM_SAMPLES_WITHOUT_UNSEEN_BUT_AUGMENTED = 20000
        self.NUM_SAMPLES_WITH_UNSEEN_BUT_AUGMENTED = 24000
        self.NUM_EVAL_DATA = 9000

    def prepare_data_module(self, **kwargs):
        datamodule = SPCUP22DataModule(
            self.BATCH_SIZE,
            dataset_root=str(ROOT.joinpath("data", "raw_audio", "spcup22")),
            **kwargs
        )
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def test_datamodule_without_unseen_data(self):
        datamodule = self.prepare_data_module()

        assert hasattr(datamodule, "train_data")
        assert hasattr(datamodule, "val_data")
        assert hasattr(datamodule, "test_data")

        annotations_df = datamodule.data.annotations_df
        labels = annotations_df["algorithm"].unique()

        self.assertEqual(len(labels), datamodule.num_classes)
        self.assertEqual(len(annotations_df), self.NUM_SAMPLES_WITHOUT_UNSEEN)

        datamodule.teardown()

    def test_datamodule_with_unseen_data(self):
        datamodule = self.prepare_data_module(
            should_include_unseen_in_training_data=True
        )

        annotations_df = datamodule.data.annotations_df
        labels = annotations_df["algorithm"].unique()

        self.assertEqual(len(labels), datamodule.num_classes)
        self.assertEqual(len(annotations_df), self.NUM_SAMPLES_WITH_UNSEEN)

        datamodule.teardown()

    def test_datamodule_without_unseen_but_augmented_data(self):
        datamodule = self.prepare_data_module(
            should_include_unseen_in_training_data=False,
            should_include_augmented_data=True,
        )

        annotations_df = datamodule.data.annotations_df
        labels = annotations_df["algorithm"].unique()

        self.assertEqual(len(labels), datamodule.num_classes)
        self.assertEqual(
            len(annotations_df), self.NUM_SAMPLES_WITHOUT_UNSEEN_BUT_AUGMENTED
        )

        datamodule.teardown()

    def test_datamodule_with_unseen_but_augmented_data(self):
        datamodule = self.prepare_data_module(
            should_include_unseen_in_training_data=True,
            should_include_augmented_data=True,
        )

        annotations_df = datamodule.data.annotations_df
        labels = annotations_df["algorithm"].unique()

        self.assertEqual(len(labels), datamodule.num_classes)
        self.assertEqual(
            len(annotations_df), self.NUM_SAMPLES_WITH_UNSEEN_BUT_AUGMENTED
        )

        datamodule.teardown()

    def test_train_data_shape(self):
        datamodule = self.prepare_data_module()

        sample, label = datamodule.train_data.__getitem__(0)
        self.assertEqual(sample.shape, (1, 96000))
        self.assertNotEqual(label, None)

        sample, label = datamodule.test_data.__getitem__(0)
        self.assertEqual(sample.shape, (1, 96000))
        self.assertNotEqual(label, None)

        sample, label = datamodule.val_data.__getitem__(0)
        self.assertEqual(sample.shape, (1, 96000))
        self.assertNotEqual(label, None)

        datamodule.teardown()

    def test_eval_data_shape(self):
        datamodule = self.prepare_data_module(should_load_eval_data=True)

        self.assertEqual(hasattr(datamodule, "train_data"), False)
        self.assertEqual(hasattr(datamodule, "val_data"), False)
        self.assertEqual(hasattr(datamodule, "test_data"), True)

        self.assertEqual(
            len(datamodule.test_data.annotations_df), self.NUM_EVAL_DATA
        )

        sample, label, filepath = datamodule.test_data.__getitem__(0)
        self.assertEqual(sample.shape, (1, 96000))
        self.assertEqual(label, -1)
        self.assertTrue(str(filepath).endswith(".wav"))

        datamodule.teardown()


if __name__ == "__main__":
    unittest.main()
