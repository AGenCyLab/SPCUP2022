import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas


class _ResBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class _ResNet(pl.LightningModule):
    def __init__(
        self,
        repeat,
        in_channels=1,
        resblock=_ResBlock,
        useBottleneck=False,
        num_classes=6,
    ):
        super(_ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module(
            "conv2_1", resblock(filters[0], filters[1], downsample=False)
        )
        for i in range(1, repeat[0]):
            self.layer1.add_module(
                "conv2_%d" % (i + 1,),
                resblock(filters[1], filters[1], downsample=False),
            )

        self.layer2 = nn.Sequential()
        self.layer2.add_module(
            "conv3_1", resblock(filters[1], filters[2], downsample=True)
        )
        for i in range(1, repeat[1]):
            self.layer2.add_module(
                "conv3_%d" % (i + 1,),
                resblock(filters[2], filters[2], downsample=False),
            )

        self.layer3 = nn.Sequential()
        self.layer3.add_module(
            "conv4_1", resblock(filters[2], filters[3], downsample=True)
        )
        for i in range(1, repeat[2]):
            self.layer3.add_module(
                "conv2_%d" % (i + 1,),
                resblock(filters[3], filters[3], downsample=False),
            )

        self.layer4 = nn.Sequential()
        self.layer4.add_module(
            "conv5_1", resblock(filters[3], filters[4], downsample=True)
        )
        for i in range(1, repeat[3]):
            self.layer4.add_module(
                "conv3_%d" % (i + 1,),
                resblock(filters[4], filters[4], downsample=False),
            )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            torch.nn.Linear(filters[4], num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class CNNs(pl.LightningModule):
    def __init__(
        self,
        network="",
        num_classes=6,
        learning_rate=1e-5,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        return_features: bool = False,
    ):
        """CNNs contains implimentation of VGG16, ResNet34, ResNet18.
        This also includes implimentation for class functions from LightningModule module.

        Args:
            network (str, optional): Network name in string. Defaults to "".
            num_classes (int, optional): Num of classes for the final layer. Defaults to 6.
            learning_rate (_type_, optional): learning rate of the optimizer. Defaults to 1e-5.
            lr_scheduler_factor (float, optional): Factor is used in lr scheduler to decrese learning rate over time. Defaults to 0.1.
            lr_scheduler_patience (int, optional): Patince is used in lr scheduler to monitor performence on given loss. Defaults to 10.
            return_features (bool, optional): Whether to include output from layer before final layer. Defaults to False.

        Raises:
            Exception: if the network name is not given or not found.
        """
        super(CNNs, self).__init__()\

        networks = ["VGG16", "ResNet34", "ResNet18"]

        self.network = network
        self.return_features = return_features

        if self.network == "VGG16":
            self.net = models.vgg16_bn()

            for p in self.net.parameters():
                p.requires_grad = False

            self.net.features[0] = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            self.net.classifier[6] = nn.Linear(
                in_features=4096, out_features=num_classes, bias=True
            )

        elif self.network == "ResNet34":
            self.net = _ResNet([3, 4, 6, 3])

        elif self.network == "ResNet18":
            self.net = _ResNet([2, 2, 2, 2])

        else:
            raise Exception(f"Use one of the followings {networks}")

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            verbose=False,
        )

    def forward(self, x):
        features = None
        if self.network == "VGG16":
            x = self.net.features(x)
            x = self.net.avgpool(x)
            x = nn.Flatten(1)(x)
            if self.return_features:
                features = self.net.classifier[:4](x)
            logits = self.net.classifier(x)
        else:
            logits = torch.sigmoid(self.net(x))

        return logits, features

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x.float())
        correct_prediction = (torch.argmax(logits, 1) == y).sum()

        loss = F.cross_entropy(logits, y)

        return {
            "loss": loss,
            "correct": correct_prediction,
            "total": len(x),
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.Tensor(
            [output["loss"] for output in outputs]
        ).mean()
        correct = torch.Tensor([output["correct"] for output in outputs]).sum()
        total = torch.Tensor([output["total"] for output in outputs]).sum()
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", correct / total, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits, _ = self(x.float())
            correct_prediction = (torch.argmax(logits, 1) == y).sum()
            loss = F.cross_entropy(logits, y)

            return {
                "loss": loss,
                "correct": correct_prediction,
                "total": len(x),
            }

    def validation_epoch_end(self, outputs):
        val_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        correct = torch.Tensor([output["correct"] for output in outputs]).sum()
        total = torch.Tensor([output["total"] for output in outputs]).sum()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", correct / total, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits, _ = self(x.float())
            correct_prediction = (torch.argmax(logits, 1) == y).sum()
            loss = F.cross_entropy(logits, y)

            return {
                "loss": loss,
                "correct": correct_prediction,
                "total": len(x),
            }

    def test_epoch_end(self, outputs):
        test_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        correct = torch.Tensor([output["correct"] for output in outputs]).sum()
        total = torch.Tensor([output["total"] for output in outputs]).sum()
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", correct / total, prog_bar=True)

        return {
            "test_loss": test_loss,
            "test_acc": correct / total,
        }

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs, _, filepaths = batch
            logits, features = self.forward(inputs.float())

            return logits, filepaths, features

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }
