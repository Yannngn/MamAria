from typing import Union

import lightning as pl
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, ListConfig
from torch import nn
from torchvision.transforms import functional as F

from src.utils.utils import load_obj

# Differences from original UNET:
# 1) Add padding to preserve original input size
# 2) Add BatchNorm2d to improve my results


class UNET(nn.Module):
    def __init__(self, config: Union[DictConfig, ListConfig]):
        super(UNET, self).__init__()

        self.min_layer_size = config.min_layer_size
        self.max_layer_size = config.max_layer_size
        mls = config.max_layer_size
        in_channels = config.in_channels
        classes = config.labels

        layers = [in_channels, config.min_layer_size]
        while mls > config.min_layer_size:
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
            [
                self.__double_conv(layer, layer // 2)
                for layer in self.layers[::-1][:-2]
            ]
        )

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(
            self.min_layer_size, classes, kernel_size=1
        )

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
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
                x = F.resize(x, concat_layer.shape[2:])

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        return self.final_conv(x)


class UNETModule(pl.LightningModule):
    def __init__(self, cfg: Union[DictConfig, ListConfig]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        self.learning_rate = self.cfg.training.params.lr

        self.model = self.get_model()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def get_model(self):
        model = UNET(self.cfg.model.params)

        return model

    def get_criterion(self):
        self.loss_fn = load_obj(self.cfg.criterion.class_name)(
            **self.cfg.criterion.params
        )

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)
        optimizer = optimizer(
            self.model.parameters(), **self.cfg.optimizer.params
        )

        lr_scheduler = load_obj(self.cfg.scheduler.class_name)
        lr_scheduler = lr_scheduler(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": self.cfg.scheduler.step,
                "monitor": self.cfg.scheduler.monitor,
            }
        ]

    def configure_metrics(self):
        self.global_metrics = {
            metrics["class_name"].split(".")[-1]: load_obj(
                metrics["class_name"]
            )(**metrics["params"])
            for metrics in self.cfg.metrics["global"]
        }
        self.label_metrics = {
            metrics["class_name"].split(".")[-1]: load_obj(
                metrics["class_name"]
            )(**metrics["params"])
            for metrics in self.cfg.metrics.label
        }

    def training_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        targets = targets.long()

        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets)

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

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.configure_metrics()

        images, targets, idx_list = batch
        targets = targets.long()

        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets)

        self.validation_dict = {
            "images": images,
            "targets": targets,
            "predictions": predictions,
            "image_idx": idx_list,
            "batch_idx": batch_idx,
        }

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(idx_list),
        )

        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        targets = self.validation_dict["targets"]
        predictions = self.validation_dict["predictions"]

        num_classes = predictions.size(dim=1)
        log_dict = {}

        for metric, f in self.global_metrics.items():
            # logging.info(f"Calculating {metric} ...")
            result = f(predictions, targets).cpu().numpy()
            if result == np.nan:
                result = 0.0
            log_dict[f"global/{metric}"] = result

        for metric, f in self.label_metrics.items():
            # logging.info(f"Calculating label {metric} ...")
            result = f(predictions, targets).cpu().numpy()
            for i in range(num_classes):
                if result[i] == np.nan:
                    result[i] = 0.0
                log_dict[f"label_{i}/{metric}"] = result[i]

        with_images = self.wandb_image_dict(log_dict)
        self.log(with_images)

        self.validation_dict.clear()

        return {"log": log_dict}

    def wandb_image_dict(self, log_dict):
        for image, target, prediction, idx in zip(
            self.validation_dict["images"],
            self.validation_dict["target"],
            self.validation_dict["predictions"],
            self.validation_dict["image_idx"],
        ):
            local_image = image.squeeze(0).cpu().numpy()
            local_target = target.cpu().numpy()
            local_prediction = prediction.cpu().numpy()

            wandb_image = wandb.Image(
                image=local_image,
                masks={
                    "prediction": {
                        "mask_data": local_prediction,
                        "class_labels": self.cfg.data.dataset.params.labels,
                    },
                    "ground truth": {
                        "mask_data": local_target,
                        "class_labels": self.cfg.data.dataset.params.labels,
                    },
                },
            )

            # lista de imagens no dict, ou imagem por imagem?
            log_dict[f"image_{idx:02d}"] = wandb_image

        return log_dict
