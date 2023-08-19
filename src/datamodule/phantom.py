from typing import Any, Callable, Literal

import albumentations as A
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from transforms.albumentations import CustomTransforms


class PhantomData(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Callable[..., Dataset],
        transforms: CustomTransforms,
        num_classes: int,
        train_image_dir: str,
        train_mask_dir: str,
        val_image_dir: str | None = None,
        val_mask_dir: str | None = None,
        test_image_dir: str | None = None,
        test_mask_dir: str | None = None,
        calib_image_dir: str | None = None,
        calib_mask_dir: str | None = None,
        num_workers: int = 2,
        batch_size: int = 16,
        val_size: float = 0.3,
    ) -> None:
        super().__init__()

        self.dataset = dataset

        # dir paths
        self.train_image_dir = train_image_dir
        self.train_mask_dir = train_mask_dir
        self.val_image_dir = val_image_dir
        self.val_mask_dir = val_mask_dir
        self.test_image_dir = test_image_dir
        self.test_mask_dir = test_mask_dir
        self.calib_image_dir = calib_image_dir
        self.calib_mask_dir = calib_mask_dir

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
        stage: Literal["fit", "test", "calibrate"] | None,
    ) -> None:
        if stage == "fit" or stage is None:
            if self.val_image_dir is None:
                self.train_dataset, self.val_dataset = self.get_train_val_dataset()
                return

            self.train_dataset = self.dataset(
                image_dir=self.train_image_dir,
                mask_dir=self.train_mask_dir,
                transforms=self.transforms.get_transforms("train"),
            )
            self.val_dataset = self.dataset(
                image_dir=self.val_image_dir,
                mask_dir=self.val_mask_dir,
                transforms=self.transforms.get_transforms("test"),
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                image_dir=self.test_image_dir,
                mask_dir=self.test_mask_dir,
                transforms=self.transforms.get_transforms("test"),
            )

        if stage == "calibrate" or stage is None:
            self.calib_dataset = self.dataset(
                image_dir=self.calib_image_dir,
                mask_dir=self.calib_mask_dir,
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

    def get_train_val_dataset(self) -> tuple[Dataset, Dataset]:
        dataset = self.dataset(
            image_dir=self.train_image_dir,
            mask_dir=self.train_mask_dir,
            transforms=self.transforms.get_transforms("train"),
        )

        indices = torch.randperm(len(dataset)).tolist()  # type: ignore
        length = int(self.val_size * len(indices))

        dataset_train = Subset(dataset, indices[length:])
        dataset_val = Subset(dataset, indices[:length])

        setattr(dataset_val, "transforms", self.transforms.get_transforms("test"))

        return dataset_train, dataset_val
