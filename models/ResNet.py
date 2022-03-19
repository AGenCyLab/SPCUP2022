import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import pandas

class ResBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResBottleneckBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(pl.LightningModule):
    def __init__(
        self,
        repeat,
        in_channels=1,
        resblock=ResBlock,
        useBottleneck=False,
        num_classes = 5,
        learning_rate = 1e-5,
        lr_scheduler_factor = 0.5,
        lr_scheduler_patience=5,
        ):
        super(ResNet,self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            torch.nn.Linear(filters[4], num_classes),
            nn.Sigmoid(),
        )
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.float())
        loss = F.cross_entropy(logits, y)
        return {
            "loss": loss,
            }
    
    def training_epoch_end(self, outputs):
        train_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        self.log("train_loss", train_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits = self(x.float())
            correct_prediction = (torch.argmax(logits, 1) == y).sum()
            loss = F.cross_entropy(logits, y)

            return {
                "loss": loss,
                "correct": correct_prediction,
                "total": len(x),
                }
    
    def training_epoch_end(self, outputs):
        val_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        correct = torch.Tensor([output["correct"] for output in outputs]).sum()
        total = torch.Tensor([output["total"] for output in outputs]).sum()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", correct/total, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            predict = self(x.float())
            predict = torch.argmax(predict)
            return {
                "label": y,
                "predict": predict.item(),
                 }
    
    def test_epoch_end(self, outputs):
        predicts = [output["predict"] for output in outputs]
        labels = [output["label"] for output in outputs]

        df = pandas.DataFrame(list(zip(labels, predicts)), columns=["label", "class"])

        df.to_csv('answer.txt', index=False, header=False)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }