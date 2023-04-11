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
                 image_size: Tuple[int] = (2816, 3584)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_size = image_size

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

        # Make only 3 labels
        # if self.threelabels is True:
        #     mask[mask == 2] = 1
        #     mask[mask == 3] = 2

        # Applying transforms (ToTensorV2)
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

        # Make only 3 labels
        # if self.threelabels is True:
        #     mask[mask == 2] = 1
        #     mask[mask == 3] = 2

        # Applying transforms (ToTensorV2)
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
              stage: Literal['training', 'testing', 'calibrating']
              ) -> None:
        if stage == 'training':
            self.train_data = self.get_dataset(
                self.cfg.dataset.params.train_image_dir,
                self.cfg.dataset.params.train_mask_dir
                )
            self.val_data = self.get_dataset(
                self.cfg.dataset.params.val_image_dir,
                self.cfg.dataset.params.val_mask_dir
                )
        if stage == 'testing':
            self.test_data = self.get_dataset(
                self.cfg.dataset.params.test_image_dir,
                self.cfg.dataset.params.test_mask_dir
                )
        if stage == 'calibrating':
            self.calib_data = self.get_dataset(
                self.cfg.dataset.params.calib_image_dir,
                self.cfg.dataset.params.calib_mask_dir
                )
            self.test_data = self.get_dataset(
                self.cfg.dataset.params.test_image_dir,
                self.cfg.dataset.params.test_mask_dir
                )

    def get_dataset(self,
                    image_dir: str,
                    mask_dir: str,
                    image_size: Tuple[int],
                    train: bool = True,
                    ) -> PhantomDCMDataset:

        if train:
            transforms = self.train_transforms()
        else:
            transforms = self.test_transforms()

        return PhantomDCMDataset(image_dir, mask_dir, transforms, image_size)

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
