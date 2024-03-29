import os
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.batch_size = 64
        self.cpu_count = c if (c := os.cpu_count()) else 8

    def setup(self, stage: str):
        # Assign datasets for use in dataloaders
        if stage == "fit":
            dataset = ImageFolder(
                self.data_dir / "train",
                transform=self.transforms
            )
            train_size = int(0.7 * len(dataset))
            valid_size = len(dataset) - train_size

            self.train_dataset, self.valid_dataset = random_split(
                dataset,
                [train_size, valid_size]
            )

        # Assign test dataset for use in dataloader's
        if stage == "test":
            self.test_dataset = ImageFolder(
                self.data_dir / "test",
                transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.cpu_count)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.cpu_count)
