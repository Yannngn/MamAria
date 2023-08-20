from typing import Any, Callable, Literal

import lightning as pl
import torch
from torch import Tensor, nn, optim
from torchvision.transforms import functional as F

from loggers.wandb_logger import WandBLogger
from src.metrics.classifier_metrics import ClassifierMetric


class UNET(nn.Module):
    """
    Differences from original UNET:
    1) Add padding to preserve original input size
    2) Add BatchNorm2d to improve my results
    """

    def __init__(
        self,
        min_layer_size: int,
        max_layer_size: int,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.min_layer_size = min_layer_size
        self.max_layer_size = mls = max_layer_size

        in_channels = in_channels
        num_classes = num_classes

        layers = [in_channels, min_layer_size]
        while mls > min_layer_size:
            layers.insert(2, mls)
            mls = int(mls * 0.5)

        self.layers = layers

        self.double_conv_downs = nn.ModuleList(
            [
                self.__double_conv(layer, layer_n)
                for layer, layer_n in zip(self.layers[:-1], self.layers[1:])
            ]
        )

        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
                for layer, layer_n in zip(
                    self.layers[::-1][:-2], self.layers[::-1][1:-1]
                )
            ]
        )

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]]
        )

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(self.min_layer_size, num_classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        return conv

    def forward(self, x):
        # down layers
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        # up layers
        for up_trans, double_conv_up, concat_layer in zip(
            self.up_trans, self.double_conv_ups, concat_layers
        ):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = F.resize(x, concat_layer.shape[2:], antialias=True)

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        return self.final_conv(x)


class UNETModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: Callable[..., optim.Optimizer],
        lr_scheduler: Callable[..., optim.lr_scheduler.LRScheduler],
        loss_fn: Callable[..., nn.Module],
        num_classes: int,
        lr: float,
        metrics: ClassifierMetric,
        weights: Tensor | None = None,
        label_map: dict[int, str] | None = None,
        monitor: str = "train_loss",
        interval: str = "epoch",
        save_dir: str = "data/results/",
    ) -> None:
        super().__init__()

        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn(weight=weights)

        self.lr = lr
        self.num_classes = num_classes

        self.metrics = metrics

        self.label_map = label_map

        self.lr_monitor = monitor
        self.lr_interval = interval
        self.save_dir = save_dir

        self.model = self.get_model()
        self.val_metrics = self.metrics.clone("val_")
        self.test_metrics = self.metrics.clone("test_")

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def get_model(self):
        return UNET(
            min_layer_size=64,
            max_layer_size=1024,
            in_channels=1,
            num_classes=self.num_classes,
        )

    def configure_optimizers(self) -> Any:
        self.optimizer = self.__optimizer(params=self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.__lr_scheduler(self.optimizer)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": self.lr_monitor,
                "interval": self.lr_interval,
            },
        }

    def training_step(
        self, batch: tuple[Tensor, Tensor, Any], batch_idx
    ) -> dict[str, Any]:
        self.model.train()
        images, targets, _ = batch
        targets = targets.long()

        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets)
        nn.CrossEntropyLoss()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(targets),
        )

        return {"loss": loss}

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Any], batch_idx
    ) -> dict[str, Any]:
        return self.__eval_step(batch, batch_idx, stage="val")

    def test_step(self, batch: tuple[Tensor, Tensor, Any], batch_idx) -> dict[str, Any]:
        return self.__eval_step(batch, batch_idx, stage="test")

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        # Predict for calibration
        self.model.eval()
        images, targets, _ = batch

        return targets, self.model(images)
        # TODO save files

    def __eval_step(
        self,
        batch: tuple[Tensor, Tensor, Any],
        batch_idx,
        stage: Literal["val", "test"],
    ) -> dict[str, Any]:
        self.model.eval()
        metrics = self.val_metrics if stage == "val" else self.test_metrics

        images, targets, (_, image_names) = batch
        targets = targets.long()

        predictions: Tensor = self.model(images)
        loss = self.loss_fn(predictions, targets)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(image_names),
        )

        metrics.update(predictions, targets)
        log_dict = metrics.compute()
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=len(targets),
        )
        metrics.reset()

        # TODO sample the images
        if isinstance(self.logger, WandBLogger):
            self.logger.prepare_logger(self.num_classes, self.label_map)

            risk = torch.argmax(predictions, dim=1).detach().cpu().numpy()

            wandb_images = self.logger.prepare_batch(
                image_names,
                images,
                targets.detach().cpu().numpy(),
                risk,
            )

            self.logger.log_image(key=stage, images=wandb_images)

        return {f"{stage}_loss": loss}
