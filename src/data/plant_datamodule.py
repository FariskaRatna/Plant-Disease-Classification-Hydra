from typing import Any, Dict, Optional, Tuple

import os
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import splitfolders

from .components.plant_dataset import PlantDataset

class PlantDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)
            ]
        )

        self.transform_val = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 8

    # def prepare_data(self) -> None:
    #     """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
    #     within a single process on CPU, so you can safely add your downloading logic within. In
    #     case of multi-node training, the execution of this hook depends upon
    #     `self.prepare_data_per_node()`.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     splitfolders.ratio(self.data_dir, seed=1337, output = 'train-test-splitted', ratio = (0.6, 0.2, 0.2))
    #     # MNIST(self.hparams.data_dir, train=True, download=True)
    #     # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_train = PlantDataset(data_dir=os.path.join(self.data_dir, 'train-test-splitted', 'train'), transform=self.transforms_train)
        self.data_test = PlantDataset(data_dir=os.path.join(self.data_dir, 'train-test-splitted', 'test'), transform=self.transforms_test)
        self.data_val = PlantDataset(data_dir=os.path.join(self.data_dir, 'train-test-splitted', 'val'), transform=self.transform_val)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = PlantDataModule()
