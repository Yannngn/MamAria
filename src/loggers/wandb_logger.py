from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor

import wandb


class WandBLogger(WandbLogger):
    def prepare_logger(self, num_classes: int, label_map: dict[int, str] | None = None):
        self.num_classes = num_classes

        if label_map is None:
            label_map = {l: str(l) for l in range(num_classes)}

        self.label_map = label_map

    def prepare_batch(
        self, image_names: list[str], images: Tensor, targets: Tensor, predictions: Tensor
    ) -> list[wandb.Image]:
        output = []
        for name, image, target, prediction in zip(image_names, images, targets, predictions):
            wandb_image = wandb.Image(
                image.squeeze(),
                caption=name,
                masks={
                    "prediction": {
                        "mask_data": prediction,
                        "class_labels": self.label_map,
                    },
                    "ground truth": {
                        "mask_data": target,
                        "class_labels": self.label_map,
                    },
                },
            )
            output.append(wandb_image)

        return output

    # TODO
    def save_collage_batch(
        self, image_names: list[str], images: Tensor, targets: Tensor, predictions: Tensor
    ) -> list[wandb.Image]:
        output = []
        for name, image, target, prediction in zip(image_names, images, targets, predictions):
            wandb_image = wandb.Image(
                image.squeeze(),
                caption=name,
                masks={
                    "prediction": {
                        "mask_data": prediction,
                        "class_labels": self.label_map,
                    },
                    "ground truth": {
                        "mask_data": target,
                        "class_labels": self.label_map,
                    },
                },
            )
            output.append(wandb_image)

        return output
