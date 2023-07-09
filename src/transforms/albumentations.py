from typing import Callable, Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


class CustomTransforms:
    """Class that stores the transforms albumentations compose object, returns the transforms depending on the stage of the run"""

    def __init__(self, compose: Callable[..., A.Compose], preprocessing: DictConfig, augmentation: DictConfig) -> None:
        self.compose = compose
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.transforms = []

        self.transforms.extend(self.preprocessing)

    def get_transforms(self, stage: Literal["train", "test"]) -> A.Compose:
        transforms = self.transforms.copy()

        if augmentation := self.augmentation.get(stage):
            transforms.extend(augmentation)

        transforms.append(ToTensorV2())

        return self.compose(transforms=transforms)
