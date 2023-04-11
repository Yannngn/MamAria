import cv2
import lightning as pl
import numpy as np
import os
import pydicom

from albumentations import Compose
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from typing import Literal, Tuple, Union

from src.utils.utils import load_obj


class PhantomDCMDataset(data.Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 transforms: Compose,
                 labels: int = 4,
                 image_size: Tuple[int] = (2816, 3584),
                 **kwargs):
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
        mask_path = os.path.join(self.mask_dir,
                                 self.images[idx]).replace("_proj.dcm",
                                                           "_risk.png")

        # Convert Image and Mask to numpy
        image = pydicom.read_file(image_path)
        image = Image.frombytes('I;16', self.image_size, image.pixel_array)
        image = np.array(image, dtype=np.float32)
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, idx


class PhantomTIFFDataset(data.Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 transforms: Compose):
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
        mask_path = os.path.join(self.mask_dir,
                                 self.images[idx].replace(".tiff",
                                                          "_mask.png"))

        # Convert Image and Mask to numpy
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB
                             ).astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class PhantomDCMData(pl.LightningDataModule):
    def __init__(self,
                 cfg: Union[DictConfig, ListConfig],
                 **kwargs) -> None:

        super().__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.dataset.params.batch_size

    def setup(self,
              stage: Literal['training', 'validating',
                             'testing', 'calibrating']
              ) -> None:
        if stage == 'training':
            self.train_data = self.get_dataset(
                self.cfg.dataset.train_image_dir,
                self.cfg.dataset.train_mask_dir,
                stage, **self.cfg.dataset.params
                )
            self.val_data = self.get_dataset(
                self.cfg.dataset.val_image_dir,
                self.cfg.dataset.val_mask_dir,
                'validating', **self.cfg.dataset.params
                )
        elif stage == 'validating':
            self.val_data = self.get_dataset(
                self.cfg.dataset.val_image_dir,
                self.cfg.dataset.val_mask_dir,
                'validating', **self.cfg.dataset.params
                )
        elif stage == 'testing':
            self.test_data = self.get_dataset(
                self.cfg.dataset.test_image_dir,
                self.cfg.dataset.test_mask_dir,
                stage, **self.cfg.dataset.params
                )
        elif stage == 'calibrating':
            self.calib_data = self.get_dataset(
                self.cfg.dataset.calib_image_dir,
                self.cfg.dataset.calib_mask_dir,
                stage, **self.cfg.dataset.params
                )
            self.test_data = self.get_dataset(
                self.cfg.dataset.test_image_dir,
                self.cfg.dataset.test_mask_dir,
                stage, **self.cfg.dataset.params
                )

    def get_dataset(self,
                    image_dir: str,
                    mask_dir: str,
                    stage: Literal['training', 'testing',
                                   'validation' 'calibrating'],
                    **kwargs,
                    ) -> PhantomDCMDataset:

        if stage == 'training':
            transforms = self.train_transforms()
        elif stage == 'validating':
            transforms = self.val_transforms()
        if stage == 'testing':
            transforms = self.test_transforms()
        else:
            transforms = self.calib_transforms()

        return PhantomDCMDataset(image_dir, mask_dir, transforms, **kwargs)

    def train_transforms(self) -> Compose:
        augs = [load_obj(aug['class_name'])(**aug['params'])
                for aug in self.cfg.augmentation.train.augs]

        bbox_params = load_obj(
            self.cfg.augmentation.train.bbox_params.class_name
            )(**self.cfg.augmentation.train.bbox_params.params)

        compose = load_obj(
            self.cfg.augmentation.compose.class_name
            )(augs, bbox_params=bbox_params)

        return compose

    def val_transforms(self) -> Compose:
        augs = [load_obj(aug['class_name'])(**aug['params'])
                for aug in self.cfg.augmentation.val.augs]

        bbox_params = load_obj(
            self.cfg.augmentation.val.bbox_params.class_name
            )(**self.cfg.augmentation.val.bbox_params.params)

        compose = load_obj(
            self.cfg.augmentation.compose.class_name
            )(augs, bbox_params=bbox_params)

        return compose

    def test_transforms(self) -> Compose:
        augs = [load_obj(aug['class_name'])(**aug['params'])
                for aug in self.cfg.augmentation.test.augs]

        bbox_params = load_obj(
            self.cfg.augmentation.test.bbox_params.class_name
            )(**self.cfg.augmentation.test.bbox_params.params)

        compose = load_obj(
            self.cfg.augmentation.compose.class_name
            )(augs, bbox_params=bbox_params)

        return compose

    def calib_transforms(self) -> Compose:
        augs = [load_obj(aug['class_name'])(**aug['params'])
                for aug in self.cfg.augmentation.calib.augs]

        bbox_params = load_obj(
            self.cfg.augmentation.calib.bbox_params.class_name
            )(**self.cfg.augmentation.calib.bbox_params.params)

        compose = load_obj(
            self.cfg.augmentation.compose.class_name
            )(augs, bbox_params=bbox_params)

        return compose

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.cfg.dataset.params.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.cfg.dataset.params.num_workers,
                          pin_memory=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.cfg.dataset.params.num_workers,
                          pin_memory=True, drop_last=True)

    def calib_dataloader(self) -> DataLoader:
        return DataLoader(self.calib_dataset,
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.cfg.dataset.params.num_workers,
                          pin_memory=True, drop_last=True)
