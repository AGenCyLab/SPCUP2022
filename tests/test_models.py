import pathlib
import sys

from tqdm import tqdm

ROOT = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT))

import unittest
from datasets.SPCUP22DataModule import SPCUP22DataModule
from models.raw_audio import IncTSSDNetWrapper, ResTSSDNetWrapper


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.BATCH_SIZE = 16
        self.datamodule = self.prepare_datamodule()

    def prepare_datamodule(self, **kwargs):
        datamodule = SPCUP22DataModule(
            self.BATCH_SIZE, str(ROOT.joinpath("data", "spcup22")), **kwargs
        )
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def get_model(self, model_type: str):
        model = None

        if model_type == "res-tssdnet":
            model = ResTSSDNetWrapper(num_classes=self.datamodule.num_classes)
        elif model_type == "inc-tssdnet":
            model = IncTSSDNetWrapper(num_classes=self.datamodule.num_classes)

        if model == None:
            raise Exception(
                "No matching model found for type '{}'".format(model_type)
            )

        model = model.cuda()
        return model

    def test_restssdnet(self):
        model = self.get_model("res-tssdnet")

        for batch in tqdm(self.datamodule.train_dataloader()):
            samples, labels = batch
            samples = samples.cuda()
            output = model(samples)

        self.assertEqual(
            output.shape, (self.BATCH_SIZE, self.datamodule.num_classes)
        )

    def test_inctssdnet(self):
        model = self.get_model("inc-tssdnet")

        for batch in tqdm(self.datamodule.train_dataloader()):
            samples, labels = batch
            samples = samples.cuda()
            output = model(samples)

        self.assertEqual(
            output.shape, (self.BATCH_SIZE, self.datamodule.num_classes)
        )


if __name__ == "__main__":
    unittest.main()
