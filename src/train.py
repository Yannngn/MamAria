import logging
import sys

import hydra
import lightning as pl
import pyrootutils
from lightning.pytorch.tuner.tuning import Tuner
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(str(ROOT_DIR))

from src.tools import (
    get_callbacks,
    get_datamodule,
    get_loggers,
    get_model_checkpoint,
    get_model_module,
    get_profiler,
    get_wandb_experiment,
    get_weights,
    setup,
)


@hydra.main(config_path=f"{ROOT_DIR}/config", config_name="testing_mnist.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    setup(cfg)

    data_module = get_datamodule(cfg, stage="fit")
    weights = get_weights(cfg, data_module=data_module)
    model_module = get_model_module(cfg, weights)
    loggers = get_loggers(cfg)
    callbacks = get_callbacks(cfg)
    profiler = get_profiler(cfg)

    if experiment := get_wandb_experiment(loggers):
        experiment.tags += OmegaConf.to_object(cfg.tags)
        experiment.config(OmegaConf.to_object(cfg))

    trainer = pl.Trainer(**cfg.trainer, profiler=profiler, logger=loggers, callbacks=callbacks)

    tuner = Tuner(trainer)

    tuner.lr_find(model_module, datamodule=data_module)
    tuner.scale_batch_size(model_module, datamodule=data_module)

    logging.info("Trainer was set. Fitting")

    trainer.fit(model_module, data_module, ckpt_path=cfg.ckpt_path)
    logging.info("Fit complete.")

    if model_checkpoint := get_model_checkpoint(callbacks):
        best_ckpt = model_checkpoint.best_model_path
        ckpt_score = model_checkpoint.best_model_score

        logging.info(f"Best checkpoint was saved at: {best_ckpt} with score: {ckpt_score:.3f}")
    else:
        logging.info("No ModelCheckpoint to test, exiting.")

        return

    if experiment:
        logging.info("Logging best checkpoint")
        experiment.log({"best_checkpoint", best_ckpt})


if __name__ == "__main__":
    train()
    logging.info("Done.")
