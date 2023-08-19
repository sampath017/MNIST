import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.batch_size = 64

    def prepare_data(self):
        # download (ontime process)
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign datasets for use in dataloaders
        if stage == "fit":
            dataset = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform
            )
            train_size = int(0.7 * len(dataset))
            valid_size = len(dataset) - train_size

            self.train_dataset, self.valid_dataset = random_split(
                dataset,
                [train_size, valid_size]
            )

        # Assign test dataset for use in dataloader's
        if stage == "test":
            self.test_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
