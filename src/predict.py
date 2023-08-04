import logging
import sys

import hydra
import lightning as pl
import pyrootutils
import wandb
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(str(ROOT_DIR))

from src.tools import get_datamodule, get_model_module, set_comet_api_key, setup


@hydra.main(config_path=f"{ROOT_DIR}/config", config_name="testing_penn.yaml", version_base=None)
def predict(cfg: DictConfig) -> None:
    cfg = setup(cfg)

    data_module = get_datamodule(cfg, stage="test")

    model_module = get_model_module(cfg)

    profiler = SimpleProfiler()

    trainer = pl.Trainer(**cfg.trainer, profiler=profiler)
    logging.info("Trainer was set. Predicting")

    logging.info("Predict:")

    trainer.predict(
        model_module,
        data_module.test_dataloader(),
        ckpt_path=cfg.ckpt_path,
    )

    logging.info("Done.")


if __name__ == "__main__":
    predict()
