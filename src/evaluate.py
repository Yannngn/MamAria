import logging
import sys

import hydra
import lightning as pl
import pyrootutils
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig, OmegaConf

import wandb

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(str(ROOT_DIR))

from src.tools import (
    get_datamodule,
    get_loggers,
    get_model_module,
    get_wandb_experiment,
    setup,
)

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(config_path=f"{ROOT_DIR}/config", config_name="testing_penn.yaml", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    cfg = setup(cfg)

    data_module = get_datamodule(cfg, stage="test")

    model_module = get_model_module(cfg)

    loggers = get_loggers(cfg)
    experiment = get_wandb_experiment(loggers)

    if experiment := get_wandb_experiment(loggers):
        experiment.tags += OmegaConf.to_object(cfg.tags)
        experiment.config(OmegaConf.to_object(cfg))

    profiler = SimpleProfiler()

    trainer = pl.Trainer(**cfg.trainer, logger=loggers, profiler=profiler)
    logging.info("Trainer was set. Fitting")

    logging.info("Evaluate:")
    trainer.test(model_module, data_module, ckpt_path=cfg.ckpt_path)

    logging.info("Done.")


if __name__ == "__main__":
    evaluate()
