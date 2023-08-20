import os
from typing import Callable, Literal

import lightning as pl
from torch.utils.data import DataLoader, Dataset

from transforms.albumentations import CustomTransforms


class PhantomData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: Callable[..., Dataset],
        transforms: CustomTransforms,
        num_classes: int,
        num_workers: int = 2,
        batch_size: int = 16,
        val_size: float = 0.3,
    ) -> None:
        super().__init__()

        self.dataset = dataset

        # dir path
        self.data_dir = data_dir

        # transforms object
        self.transforms = transforms

        # params
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size

        # size of train_val split
        self.val_size = val_size

    def setup(
        self,
        stage: Literal["fit", "test", "calib"] | None,
    ) -> None:
        if stage in ["fit", None]:
            self.train_dataset = self.dataset(
                data_dir=os.path.join(self.data_dir, "train"),
                transforms=self.transforms.get_transforms("train"),
            )
            self.val_dataset = self.dataset(
                data_dir=os.path.join(self.data_dir, "val"),
                transforms=self.transforms.get_transforms("test"),
            )

        if stage in ["test", None]:
            self.test_dataset = self.dataset(
                data_dir=os.path.join(self.data_dir, "test"),
                transforms=self.transforms.get_transforms("test"),
            )

        if stage in ["calib", None]:
            self.calib_dataset = self.dataset(
                data_dir=os.path.join(self.data_dir, "calib"),
                transforms=self.transforms.get_transforms("test"),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def calib_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
