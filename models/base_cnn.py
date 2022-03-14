import sys
import pathlib

import torch

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)


import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """
    Base CNN:
    conv2d -> batchNorm2d -> maxPool2d
    """
    def __init__(self, input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(BaseCNN, self).__init__()
        self.conv = nn.Conv2d(input_dim,
                              output_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              )
        self.bn = nn.BatchNorm2d(output_dim)
        self.pool = nn.MaxPool2d(input_dim, stride=stride)

class BaseCNNWrapper(pl.LightningModule):
    def __init__(self,
                 num_classes: int = 5,
                 learning_rate: float = 1e-3,
                 exp_lr_scheduler_gamma: float = 0.95,
                 ) -> None:
        super(BaseCNNWrapper, self).__init__()
        self.base_cnn = BaseCNN()
        self.learning_rate = learning_rate
        self.exp_lr_scheduler_gamma = exp_lr_scheduler_gamma
        self.final_layer = nn.Linear(32, num_classes)

        return

    def forward(self, x):
        model = self.base_cnn
        x = model.conv(x)
        x = model.bn(x)
        x = model.pool(x)
        x = F.relu(x)

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

