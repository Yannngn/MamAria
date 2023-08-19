import os
from typing import Any

import cv2
import numpy as np
import pydicom
from albumentations import Compose
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class PhantomDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Any | None = None,
        transforms: Compose | None = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        self.image_size = image_size

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        self.image_names = [os.path.splitext(os.path.basename(image))[0] for image in self.images]

    def __len__(self):
        return len(self.images)


class PhantomDCMDataset(PhantomDataset):
    def __post_init__(self):
        self.image_size = self.image_size if self.image_size is not None else (2816, 3584)

    def __getitem__(self, idx: int) -> tuple[Any, Any, tuple[int, str]]:
        # Get Image and corresponding Mask paths
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]).replace("_proj.dcm", "_risk.png")
        # Convert Image and Mask to numpy
        image = pydicom.read_file(image_path)
        image = Image.frombytes("I;16", tuple(self.image_size), image.pixel_array)
        image = np.array(image, dtype=np.float32)
        mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        return image, mask, (idx, self.image_names[idx])


class PhantomTIFFDataset(PhantomDataset):
    def __getitem__(self, idx: int) -> tuple[Any, Any, tuple[int, str]]:
        # Get Image and corresponding Mask paths
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]).replace(".tiff", "_mask.png")

        # Convert Image and Mask to numpy
        image = np.ndarray(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        mask = np.ndarray(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, (idx, self.image_names[idx])
