from inspect import Parameter
import sys
import pathlib
from typing import Iterator, Optional

import torch

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from models.tssd_net.models import SSDNet1D as ResTSSDNet
from models.tssd_net.models import DilatedNet as IncTSSDNet


def get_optimizers(
    parameters: Iterator[Parameter],
    learning_rate: float = 1e-3,
    exp_lr_scheduler_gamma: float = 0.95,
):
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=exp_lr_scheduler_gamma
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": "val_loss",
    }


class ResTSSDNetWrapper(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-3,
        exp_lr_scheduler_gamma: float = 0.95,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.res_tssd_net = ResTSSDNet()
        self.learning_rate = learning_rate
        self.exp_lr_scheduler_gamma = exp_lr_scheduler_gamma
        self.final_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        model = self.res_tssd_net

        x = F.relu(model.bn1(model.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = model.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = model.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = model.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = model.RSM4(x)
        # x = F.max_pool1d(x, kernel_size=x.shape[-1])
        x = F.max_pool1d(x, kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(model.fc1(x))
        x = F.relu(model.fc2(x))
        logits = self.final_layer(x)

        return logits

    def configure_optimizers(self):
        return get_optimizers(
            self.parameters(), self.learning_rate, self.exp_lr_scheduler_gamma
        )

    def training_step(self, batch, batch_idx: int):
        inputs, labels = batch
        logits = self.forward(inputs)

        loss = F.cross_entropy(logits, labels)

        self.log("train_loss", loss.item())
        self.log("epoch", self.current_epoch)

        return loss

    def validation_step(self, val_batch, batch_idx: int):
        with torch.no_grad():
            inputs, labels = val_batch
            logits = self.forward(inputs)

            loss = F.cross_entropy(logits, labels)

            self.log("val_loss", loss.item())

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None,
    ):
        with torch.no_grad():
            inputs, _, filepaths = batch
            logits = self.forward(inputs)
            return logits, filepaths


class IncTSSDNetWrapper(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-3,
        exp_lr_scheduler_gamma: float = 0.95,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.inc_tssd_net = IncTSSDNet()
        self.learning_rate = learning_rate
        self.exp_lr_scheduler_gamma = exp_lr_scheduler_gamma
        self.final_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        model = self.inc_tssd_net

        x = F.relu(model.bn1(model.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        x = F.max_pool1d(model.DCM1(x), kernel_size=4)
        x = F.max_pool1d(model.DCM2(x), kernel_size=4)
        x = F.max_pool1d(model.DCM3(x), kernel_size=4)
        # x = F.max_pool1d(model.DCM4(x), kernel_size=x.shape[-1])
        x = F.max_pool1d(model.DCM4(x), kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(model.fc1(x))
        x = F.relu(model.fc2(x))
        logits = self.final_layer(x)

        return logits

    def configure_optimizers(self):
        return get_optimizers(
            self.parameters(), self.learning_rate, self.exp_lr_scheduler_gamma
        )

    def training_step(self, batch, batch_idx: int):
        inputs, labels = batch
        logits = self.forward(inputs)

        loss = F.cross_entropy(logits, labels)

        self.log("train_loss", loss.item())
        self.log("epoch", self.current_epoch)

        return loss

    def validation_step(self, val_batch, batch_idx: int):
        with torch.no_grad():
            inputs, labels = val_batch
            logits = self.forward(inputs)

            loss = F.cross_entropy(logits, labels)

            self.log("val_loss", loss.item())

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None,
    ):
        with torch.no_grad():
            inputs, _, filepaths = batch
            logits = self.forward(inputs)
            return logits, filepaths
