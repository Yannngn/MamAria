import logging
import os
import random
from typing import Literal

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.profilers import Profiler
from omegaconf import DictConfig
from torch import Tensor
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from .utils import log, log_list


@log
def get_datamodule(cfg: DictConfig, stage: Literal["fit", "test"] | None = None) -> pl.LightningDataModule:
    """returns configured data module"""

    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    data_module.setup(stage)  # type: ignore

    if stage in ("fit", None) and getattr(cfg.trainer, "log_every_n_steps", None) is None:
        steps_per_epoch = len(data_module.val_dataset) // cfg.datamodule.batch_size  # type: ignore
        setattr(cfg.trainer, "log_every_n_steps", steps_per_epoch)

    return data_module


@log
def get_weights(cfg: DictConfig, data_module: pl.LightningDataModule):
    weights = torch.zeros(cfg.num_classes, dtype=torch.float32)

    multiplier = torch.tensor(cfg.multiplier) if cfg.get("multiplier", None) else torch.ones_like(weights)

    loader = data_module.train_dataloader()
    total_pixels = None

    for batch in loader:
        _, masks, _ = batch

        if not total_pixels:
            shape = masks[0].shape
            total_pixels = len(data_module.train_dataset) * shape[-1] * shape[-2]  # type: ignore

        for label in range(cfg.num_classes):
            weights[label] += torch.sum(masks == label)

    weights = torch.divide(multiplier, weights / total_pixels)

    # weights = torch.divide(multiplier, 1 + torch.exp(-weights / (1 - total_pixels)))

    return weights  # [  1.7175,   4.1301,   5.8554, 206.4117]


@log
def get_model_module(cfg: DictConfig, weights: Tensor | None = None) -> pl.LightningModule:
    """returns configured model module"""
    model_module = hydra.utils.instantiate(cfg.module)
    model_module = model_module(weights=weights)
    torch.compile(model_module)

    return model_module


@log_list
def get_loggers(cfg: DictConfig) -> list[Logger]:
    """return list of loggers defined in config"""
    if loggers_config := cfg.get("loggers"):
        return [hydra.utils.instantiate(logger) for logger in loggers_config]

    return [CSVLogger(cfg.paths.log_dir)]


# FIXME TypeError: 'NoneType' object is not iterable
@log
def get_wandb_experiment(loggers: list[Logger]) -> Run | RunDisabled | None:
    """returns the comet experiment from the list of loggers if it exists"""

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            return logger.experiment


@log_list
def get_callbacks(cfg: DictConfig) -> list[Callback]:
    """returns list of callbacks defined in config"""

    if callbacks_config := cfg.get("callbacks"):
        callbacks = []
        for callback in callbacks_config:
            if str(callback._target_).endswith(("EarlyStopping", "ModelCheckpoint")):
                # instantiate earlystopping and modelcheckpoint with monitor
                callbacks.append(hydra.utils.instantiate(callback, monitor=cfg.module.monitor))
                continue

            callbacks.append(hydra.utils.instantiate(callback))

        return callbacks

    return [Timer()]


@log
def get_model_checkpoint(callbacks: list[Callback]) -> ModelCheckpoint | None:
    """returns the model_checkpoint callback if in list of callbacks"""

    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


@log
def get_profiler(cfg: DictConfig) -> Profiler | None:
    return hydra.utils.instantiate(cfg.profiler)


def setup(cfg: DictConfig) -> DictConfig:
    """makes log_dir, initializes logging, seeds run and checks train data csv file"""

    if getattr(cfg, "seed") is None:
        # generates a 32bit random number to be used as seed
        setattr(cfg, "seed", random.randint(1, 2**32 - 1))

    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(cfg.paths.log_dir, cfg.paths.logging_file),
        encoding="utf-8",
        level=logging.INFO,
    )

    logging.info(f"Seed set to {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    logging.info(f"Torch set to medium matmul_precision")
    torch.set_float32_matmul_precision("medium")

    return cfg
