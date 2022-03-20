import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas

class CNNs(pl.LightningModule):
    def __init__(
        self,
        network="VGG16",
        num_classes = 6,
        learning_rate = 1e-5,
        lr_scheduler_factor = 0.5,
        lr_scheduler_patience=5,
        ):
        super(CNNs,self).__init__()

        networks = ["VGG16", "ResNet34", "ResNet18"]

        if network=="VGG16":
            self.net = models.vgg16_bn()

            for p in self.net.parameters():
                p.requires_grad=False

            self.net.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

        elif network=="ResNet34":
            self.net = models.resnet34()

            for p in self.net.parameters():
                p.requires_grad=False

            self.net.conv1 =  nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.net.fc =  nn.Linear(in_features=512, out_features=num_classes, bias=True)

        elif network=="ResNet18":
            self.net = models.resnet18()

            for p in self.net.parameters():
                p.requires_grad=False

            self.net.conv1 =  nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.net.fc =  nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            raise Exception(f"Use one of the followings {networks}")
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx):
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
        train_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        correct = torch.Tensor([output["correct"] for output in outputs]).sum()
        total = torch.Tensor([output["total"] for output in outputs]).sum()
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", correct/total, prog_bar=True)

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
    
    def validation_epoch_end(self, outputs):
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