import sys
import pathlib
from typing import Optional

import torch

ROOT = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT)

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from models.tssd_net.models import DilatedNet as IncTSSDNet


class IncTSSDNetWrapper(pl.LightningModule):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.inc_tssd_net = IncTSSDNet()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val_loss",
        }

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
            inputs, labels = batch
            logits = self.forward(inputs)
            return logits
