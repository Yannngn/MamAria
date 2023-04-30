import os
from typing import Literal, Tuple, Union

import cv2
import lightning as pl
import numpy as np
import pydicom
from albumentations import Compose
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

from src.utils.utils import load_obj


class PhantomDCMDataset(data.Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transforms: Compose,
        labels: int = 4,
        image_size: Tuple[int] = (2816, 3584),
        **kwargs,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_size = image_size
        self.labels = labels
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get Image and corresponding Mask paths
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]).replace(
            "_proj.dcm", "_risk.png"
        )

        # Convert Image and Mask to numpy
        image = pydicom.read_file(image_path)
        image = Image.frombytes("I;16", self.image_size, image.pixel_array)
        image = np.array(image, dtype=np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, idx


class PhantomTIFFDataset(data.Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transforms: Compose):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get Image and corresponding Mask paths
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".tiff", "_mask.png")
        )

        # Convert Image and Mask to numpy
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class PhantomData(pl.LightningDataModule):
    def __init__(self, cfg: Union[DictConfig, ListConfig], **kwargs) -> None:
        super().__init__()

        self.cfg = cfg

    def setup(
        self,
        stage: Literal["training", "validating", "testing", "calibrating"],
    ) -> None:
        if stage == "training":
            self.train_data = self.get_dataset(
                self.cfg.data.train_image_dir,
                self.cfg.data.train_mask_dir,
                stage,
                **self.cfg.data.dataset.params,
            )
            self.val_data = self.get_dataset(
                self.cfg.data.val_image_dir,
                self.cfg.data.val_mask_dir,
                "validating",
                **self.cfg.data.dataset.params,
            )
        elif stage == "validating":
            self.val_data = self.get_dataset(
                self.cfg.data.val_image_dir,
                self.cfg.data.val_mask_dir,
                stage,
                **self.cfg.data.dataset.params,
            )
        elif stage == "testing":
            self.test_data = self.get_dataset(
                self.cfg.data.test_image_dir,
                self.cfg.data.test_mask_dir,
                stage,
                **self.cfg.data.dataset.params,
            )
        elif stage == "calibrating":
            self.calib_data = self.get_dataset(
                self.cfg.data.calib_image_dir,
                self.cfg.data.calib_mask_dir,
                stage,
                **self.cfg.data.dataset.params,
            )
            self.test_data = self.get_dataset(
                self.cfg.data.test_image_dir,
                self.cfg.data.test_mask_dir,
                stage,
                **self.cfg.data.dataset.params,
            )

    def get_dataset(
        self,
        image_dir: str,
        mask_dir: str,
        stage: Literal["training", "testing", "validation", "calibrating"],
        **kwargs,
    ) -> data.Dataset:
        if stage == "training":
            transforms = self.transforms(True)
        else:
            transforms = self.transforms(False)

        return load_obj(self.cfg.data.dataset.class_name)(
            image_dir, mask_dir, transforms, **kwargs
        )

    def transforms(self, training=True) -> Compose:
        transforms = []
        transforms.extend(
            [
                load_obj(step["class_name"])(**step["params"])
                for step in self.cfg.transforms.preprocessing
            ]
        )
        if training:
            transforms.extend(
                [
                    load_obj(aug["class_name"])(**aug["params"])
                    for aug in self.cfg.transforms.augs
                ]
            )

        transforms.extend(
            [
                load_obj(step["class_name"])(**step["params"])
                for step in self.cfg.transforms.final
            ]
        )

        compose = load_obj(self.cfg.augmentation.compose.class_name)(
            transforms
        )

        return compose

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.cfg.data.params.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.cfg.data.params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.cfg.data.params.num_workers,
        )

    def calib_dataloader(self) -> DataLoader:
        return DataLoader(
            self.calib_dataset,
            shuffle=False,
            **self.cfg.data.params.num_workers,
        )
