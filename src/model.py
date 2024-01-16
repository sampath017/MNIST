import torch

from torchmetrics.functional import accuracy
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn

model_cnn = nn.Sequential(
    # Feature extractor
    nn.LazyConv2d(32, kernel_size=3),
    nn.ReLU(),
    nn.LazyConv2d(32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.25),

    nn.LazyConv2d(64, kernel_size=3),
    nn.ReLU(),
    nn.LazyConv2d(64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.25),

    nn.LazyConv2d(128, kernel_size=3),
    nn.ReLU(),
    nn.LazyConv2d(128, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.25),

    # learner
    nn.Flatten(),

    nn.LazyLinear(256),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.LazyLinear(10)
)


class Digits(LightningModule):
    def __init__(self, optimizer_name, optimizer_hparams={}):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cnn

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task='multiclass', num_classes=10)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc*100.0, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task='multiclass', num_classes=10)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc*100.0, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task='multiclass', num_classes=10)

        self.log("test_loss", loss)
        self.log("test_acc", acc*100.0)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'SGD':  # type: ignore
            optimizer = torch.optim.SGD(
                self.parameters(), **self.hparams.optimizer_hparams)  # type: ignore

        elif self.hparams.optimizer_name == 'Adam':  # type: ignore
            optimizer = torch.optim.Adam(
                self.parameters(), **self.hparams.optimizer_hparams)  # type: ignore
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'  # type: ignore # nopep8

        return optimizer
